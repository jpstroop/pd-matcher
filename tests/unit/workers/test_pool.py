"""Tests for :mod:`pd_matcher.workers.pool` (non-spawn paths)."""

from multiprocessing import Event
from multiprocessing import get_context
from multiprocessing.queues import Queue as MpQueue
from os import environ
from pathlib import Path
from queue import Full

from msgspec.json import decode as json_decode
from pytest import MonkeyPatch
from pytest import raises

from pd_matcher.config.schemas import MatchingConfig
from pd_matcher.config.schemas import PairingConfig
from pd_matcher.index.lookup import NyplIndexLookup
from pd_matcher.match.idf import IdfTable
from pd_matcher.match.idf import build_author_idf_table
from pd_matcher.match.idf import build_idf_table
from pd_matcher.match.idf import build_publisher_idf_table
from pd_matcher.match.prepare import prepare_marc
from pd_matcher.match.result import MatchResult
from pd_matcher.models import MarcRecord
from pd_matcher.output.jsonl_writer import JsonlResultWriter
from pd_matcher.parsers.marc import iter_marc_records
from pd_matcher.workers.events import decode_stats_event
from pd_matcher.workers.messages import WorkerOutput
from pd_matcher.workers.messages import encode_worker_output
from pd_matcher.workers.pool import RunReport
from pd_matcher.workers.pool import WorkerDiedError
from pd_matcher.workers.pool import _build_jsonl_writer
from pd_matcher.workers.pool import _dead_workers
from pd_matcher.workers.pool import _default_workers
from pd_matcher.workers.pool import _drain_sentinels
from pd_matcher.workers.pool import _make_guarded_put
from pd_matcher.workers.pool import _resolve_source
from pd_matcher.workers.pool import _shutdown_predicate
from pd_matcher.workers.pool import _terminate_if_alive
from pd_matcher.workers.pool import _worker_entry
from pd_matcher.workers.pool import _writer_entry
from pd_matcher.workers.pool import run_match
from pd_matcher.workers.producer import encode_batch

_FIXTURES = Path(__file__).resolve().parents[2] / "fixtures"


def test_default_workers_returns_positive_count() -> None:
    assert _default_workers() >= 1


def test_build_jsonl_writer_returns_jsonl_writer(tmp_path: Path) -> None:
    writer = _build_jsonl_writer(tmp_path / "x.jsonl")
    assert isinstance(writer, JsonlResultWriter)
    assert writer._matches_only is False


def test_build_jsonl_writer_honors_matches_only(tmp_path: Path) -> None:
    writer = _build_jsonl_writer(tmp_path / "x.jsonl", matches_only=True)
    assert isinstance(writer, JsonlResultWriter)
    assert writer._matches_only is True


def test_shutdown_predicate_tracks_event() -> None:
    event = Event()
    predicate = _shutdown_predicate(event)
    assert predicate() is False
    event.set()
    assert predicate() is True


def test_worker_entry_runs_in_process_against_real_queues(
    tiny_index_path: Path,
    tiny_idf: IdfTable,
    tiny_author_idf: IdfTable,
    tiny_publisher_idf: IdfTable,
    matching_config: MatchingConfig,
    pairing_config: PairingConfig,
) -> None:
    """``_worker_entry`` is reachable in-process via real spawn-context queues."""
    ctx = get_context("spawn")
    input_queue: MpQueue[bytes | None] = ctx.Queue()
    output_queue: MpQueue[bytes | None] = ctx.Queue()
    stats_queue: MpQueue[bytes] = ctx.Queue()
    event = ctx.Event()
    marc = MarcRecord(control_id="m", title="t", title_main="t", publication_year=1940)
    input_queue.put(encode_batch((marc,)))
    input_queue.put(None)
    _worker_entry(
        index_path=tiny_index_path,
        matching_config=matching_config,
        pairing_config=pairing_config,
        idf=tiny_idf,
        author_idf=tiny_author_idf,
        publisher_idf=tiny_publisher_idf,
        calibrator=None,
        learned_model_dir=None,
        input_queue=input_queue,
        output_queue=output_queue,
        stats_queue=stats_queue,
        shutdown_event=event,
    )
    output_blob = output_queue.get(timeout=2.0)
    assert output_blob is not None
    stats_blob = stats_queue.get(timeout=2.0)
    assert decode_stats_event(stats_blob)


def test_writer_entry_runs_in_process_against_real_queues(tmp_path: Path) -> None:
    """``_writer_entry`` writes through a real CSV when fed a single payload."""
    ctx = get_context("spawn")
    output_queue: MpQueue[bytes | None] = ctx.Queue()
    stats_queue: MpQueue[bytes] = ctx.Queue()
    event = ctx.Event()
    marc = MarcRecord(control_id="m", title="t", title_main="t")
    payload = WorkerOutput(
        marc=marc,
        match=MatchResult(
            marc_control_id="m",
            best=None,
            alternates=(),
            candidates_considered=0,
        ),
        matched_nypl=None,
    )
    output_queue.put(encode_worker_output(payload))
    output_queue.put(None)
    path = tmp_path / "out.jsonl"
    _writer_entry(
        output_path=path,
        writer_factory=JsonlResultWriter,
        output_queue=output_queue,
        stats_queue=stats_queue,
        shutdown_event=event,
    )
    records = [
        json_decode(line, type=dict[str, str])
        for line in path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    assert len(records) == 1


def test_run_match_rejects_zero_workers(
    tiny_index_path: Path,
    tiny_idf: IdfTable,
    tiny_author_idf: IdfTable,
    tiny_publisher_idf: IdfTable,
    matching_config: MatchingConfig,
    pairing_config: PairingConfig,
    tmp_path: Path,
) -> None:
    with raises(ValueError, match="workers must be >= 1"):
        run_match(
            marc_path=_FIXTURES / "tiny.marcxml",
            index_path=tiny_index_path,
            output_path=tmp_path / "results.jsonl",
            matching_config=matching_config,
            pairing_config=pairing_config,
            idf=tiny_idf,
            author_idf=tiny_author_idf,
            publisher_idf=tiny_publisher_idf,
            workers=0,
        )


def test_run_match_returns_run_report(
    pairing_config: PairingConfig,
    tmp_path: Path,
) -> None:
    """End-to-end run with default args returns a populated :class:`RunReport`."""
    from pd_matcher.index.builder import build_index

    reg_dir = tmp_path / "reg"
    ren_dir = tmp_path / "ren"
    reg_dir.mkdir()
    ren_dir.mkdir()
    (reg_dir / "tiny_reg.xml").write_bytes((_FIXTURES / "tiny_reg.xml").read_bytes())
    (ren_dir / "tiny_ren.tsv").write_bytes((_FIXTURES / "tiny_ren.tsv").read_bytes())
    index_path = tmp_path / "idx.lmdb"
    build_index(reg_dir=reg_dir, ren_dir=ren_dir, out_path=index_path)
    with NyplIndexLookup(index_path) as lookup:
        idf = build_idf_table(lookup)
        author_idf = build_author_idf_table(lookup)
        publisher_idf = build_publisher_idf_table(lookup)
    config = MatchingConfig(
        title_weight=0.50,
        author_weight=0.20,
        publisher_weight=0.10,
        edition_weight=0.05,
        lccn_weight=0.10,
        isbn_weight=0.05,
        extent_weight=0.0,
        volume_weight=0.0,
        year_window=2,
        min_combined_score=30.0,
        scorer="weighted_mean",
    )
    output_path = tmp_path / "results.jsonl"
    report = run_match(
        marc_path=_FIXTURES / "tiny.marcxml",
        index_path=index_path,
        output_path=output_path,
        matching_config=config,
        pairing_config=pairing_config,
        idf=idf,
        author_idf=author_idf,
        publisher_idf=publisher_idf,
        workers=1,
        batch_size=2,
        queue_maxsize=4,
        report_interval_seconds=0.05,
    )
    assert isinstance(report, RunReport)
    assert report.records_processed > 0
    assert report.records_written == report.records_processed
    assert report.interrupted is False
    assert output_path.exists()


def test_run_match_pins_numeric_threads_before_spawning(
    pairing_config: PairingConfig,
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    """The match pool pins numeric-lib threads in the parent env (issue #101).

    Oversubscription only bites when the worker pool runs, so the env cap must
    be applied inside :func:`run_match`. Single-process commands never call it.
    """
    from pd_matcher.workers.thread_limits import pin_numeric_threads_in_env

    for name in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        monkeypatch.delenv(name, raising=False)
    calls: list[bool] = []

    def spy() -> None:
        calls.append(True)
        pin_numeric_threads_in_env()

    monkeypatch.setattr("pd_matcher.workers.pool.pin_numeric_threads_in_env", spy)
    index_path, idf, author_idf, publisher_idf = _build_tiny_index(tmp_path)
    run_match(
        marc_path=_FIXTURES / "tiny.marcxml",
        index_path=index_path,
        output_path=tmp_path / "results.jsonl",
        matching_config=_tiny_match_config(),
        pairing_config=pairing_config,
        idf=idf,
        author_idf=author_idf,
        publisher_idf=publisher_idf,
        workers=1,
        batch_size=2,
        queue_maxsize=4,
        report_interval_seconds=0.05,
    )
    assert calls == [True]
    assert environ["OMP_NUM_THREADS"] == "1"
    assert environ["MKL_NUM_THREADS"] == "1"
    assert environ["OPENBLAS_NUM_THREADS"] == "1"
    assert environ["NUMEXPR_NUM_THREADS"] == "1"


def _build_tiny_index(tmp_path: Path) -> tuple[Path, IdfTable, IdfTable, IdfTable]:
    """Build a tiny LMDB index and its IDF tables from the shared fixtures."""
    from pd_matcher.index.builder import build_index

    reg_dir = tmp_path / "reg"
    ren_dir = tmp_path / "ren"
    reg_dir.mkdir()
    ren_dir.mkdir()
    (reg_dir / "tiny_reg.xml").write_bytes((_FIXTURES / "tiny_reg.xml").read_bytes())
    (ren_dir / "tiny_ren.tsv").write_bytes((_FIXTURES / "tiny_ren.tsv").read_bytes())
    index_path = tmp_path / "idx.lmdb"
    build_index(reg_dir=reg_dir, ren_dir=ren_dir, out_path=index_path)
    with NyplIndexLookup(index_path) as lookup:
        idf = build_idf_table(lookup)
        author_idf = build_author_idf_table(lookup)
        publisher_idf = build_publisher_idf_table(lookup)
    return index_path, idf, author_idf, publisher_idf


def _tiny_match_config() -> MatchingConfig:
    return MatchingConfig(
        title_weight=0.50,
        author_weight=0.20,
        publisher_weight=0.10,
        edition_weight=0.05,
        lccn_weight=0.10,
        isbn_weight=0.05,
        extent_weight=0.0,
        volume_weight=0.0,
        year_window=2,
        min_combined_score=30.0,
        scorer="weighted_mean",
    )


def test_run_match_matches_only_emits_no_blank_rows(
    pairing_config: PairingConfig,
    tmp_path: Path,
) -> None:
    """``matches_only`` keeps only genuine pairs; every emitted row is a match."""
    index_path, idf, author_idf, publisher_idf = _build_tiny_index(tmp_path)
    output_path = tmp_path / "results.jsonl"
    report = run_match(
        marc_path=_FIXTURES / "tiny.marcxml",
        index_path=index_path,
        output_path=output_path,
        matching_config=_tiny_match_config(),
        pairing_config=pairing_config,
        idf=idf,
        author_idf=author_idf,
        publisher_idf=publisher_idf,
        workers=1,
        batch_size=2,
        queue_maxsize=4,
        matches_only=True,
        report_interval_seconds=0.05,
    )
    rows = [
        json_decode(line, type=dict[str, str])
        for line in output_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    assert report.records_written == len(rows)
    assert all(row["match_source_id"] != "" for row in rows)


def test_run_match_uses_explicit_writer_factory(
    pairing_config: PairingConfig,
    tmp_path: Path,
) -> None:
    """An explicit ``writer_factory`` bypasses the default ``matches_only`` factory."""
    index_path, idf, author_idf, publisher_idf = _build_tiny_index(tmp_path)
    output_path = tmp_path / "results.jsonl"
    report = run_match(
        marc_path=_FIXTURES / "tiny.marcxml",
        index_path=index_path,
        output_path=output_path,
        matching_config=_tiny_match_config(),
        pairing_config=pairing_config,
        idf=idf,
        author_idf=author_idf,
        publisher_idf=publisher_idf,
        workers=1,
        batch_size=2,
        queue_maxsize=4,
        writer_factory=JsonlResultWriter,
        matches_only=True,
        report_interval_seconds=0.05,
    )
    assert report.records_written == report.records_processed


def test_resolve_source_requires_exactly_one_input(tmp_path: Path) -> None:
    with raises(ValueError, match="exactly one of"):
        _resolve_source(None, None)
    with raises(ValueError, match="exactly one of"):
        _resolve_source(tmp_path / "a.xml", tmp_path / "prepared")


def test_resolve_source_returns_file_iterator() -> None:
    records = list(_resolve_source(_FIXTURES / "tiny.marcxml", None))
    assert records == list(iter_marc_records(_FIXTURES / "tiny.marcxml"))


def test_resolve_source_returns_prepared_iterator(tmp_path: Path) -> None:
    prepared = tmp_path / "prepared"
    prepare_marc(_FIXTURES / "tiny.marcxml", prepared, chunk_size=3)
    records = list(_resolve_source(None, prepared))
    assert records == list(iter_marc_records(_FIXTURES / "tiny.marcxml"))


class _FakeProcess:
    """Stand-in for :class:`SpawnProcess` used by pool helper tests.

    ``alive_sequence`` drives successive ``is_alive`` answers so a single
    test can model the live → terminate → dead transitions deterministically;
    ``name`` and ``exitcode`` back the liveness checks in ``_dead_workers``.
    """

    __slots__ = ("_index", "alive_sequence", "calls", "exitcode", "name")

    def __init__(
        self,
        alive_sequence: list[bool] | None = None,
        *,
        name: str = "worker",
        exitcode: int | None = None,
    ) -> None:
        self.alive_sequence: list[bool] = alive_sequence if alive_sequence is not None else []
        self.calls: list[tuple[str, float | None]] = []
        self._index: int = 0
        self.name: str = name
        self.exitcode: int | None = exitcode

    def is_alive(self) -> bool:
        value = self.alive_sequence[self._index]
        self._index += 1
        return value

    def terminate(self) -> None:
        self.calls.append(("terminate", None))

    def kill(self) -> None:
        self.calls.append(("kill", None))

    def join(self, timeout: float | None = None) -> None:
        self.calls.append(("join", timeout))


class _FakePutQueue:
    """Input-queue stand-in whose bounded ``put`` raises ``Full`` on cue.

    The first ``full_before_success`` calls raise :class:`queue.Full`; every
    subsequent call records the payload. ``put`` mirrors the real
    :class:`multiprocessing.Queue` signature so it satisfies the pool's
    ``_InputQueuePut`` protocol.
    """

    __slots__ = ("_full_remaining", "puts")

    def __init__(self, full_before_success: int = 0) -> None:
        self._full_remaining: int = full_before_success
        self.puts: list[bytes | None] = []

    def put(self, obj: bytes | None, block: bool = True, timeout: float | None = None) -> None:
        if self._full_remaining > 0:
            self._full_remaining -= 1
            raise Full
        self.puts.append(obj)


def test_terminate_if_alive_noop_when_already_exited() -> None:
    process = _FakeProcess(alive_sequence=[False])
    _terminate_if_alive(process, timeout=5.0)
    assert process.calls == []


def test_terminate_if_alive_terminates_then_returns_when_join_succeeds() -> None:
    process = _FakeProcess(alive_sequence=[True, False])
    _terminate_if_alive(process, timeout=5.0)
    assert process.calls == [("terminate", None), ("join", 5.0)]


def test_terminate_if_alive_escalates_to_kill_when_terminate_does_not_take() -> None:
    process = _FakeProcess(alive_sequence=[True, True])
    _terminate_if_alive(process, timeout=5.0)
    assert process.calls == [
        ("terminate", None),
        ("join", 5.0),
        ("kill", None),
        ("join", None),
    ]


def test_run_match_consumes_prepared_chunks(
    pairing_config: PairingConfig,
    tmp_path: Path,
) -> None:
    """``run_match`` over a prepared directory matches the file-mode count."""
    from pd_matcher.index.builder import build_index

    reg_dir = tmp_path / "reg"
    ren_dir = tmp_path / "ren"
    reg_dir.mkdir()
    ren_dir.mkdir()
    (reg_dir / "tiny_reg.xml").write_bytes((_FIXTURES / "tiny_reg.xml").read_bytes())
    (ren_dir / "tiny_ren.tsv").write_bytes((_FIXTURES / "tiny_ren.tsv").read_bytes())
    index_path = tmp_path / "idx.lmdb"
    build_index(reg_dir=reg_dir, ren_dir=ren_dir, out_path=index_path)
    with NyplIndexLookup(index_path) as lookup:
        idf = build_idf_table(lookup)
        author_idf = build_author_idf_table(lookup)
        publisher_idf = build_publisher_idf_table(lookup)
    config = MatchingConfig(
        title_weight=0.50,
        author_weight=0.20,
        publisher_weight=0.10,
        edition_weight=0.05,
        lccn_weight=0.10,
        isbn_weight=0.05,
        extent_weight=0.0,
        volume_weight=0.0,
        year_window=2,
        min_combined_score=30.0,
        scorer="weighted_mean",
    )
    prepared = tmp_path / "prepared"
    prepare_report = prepare_marc(_FIXTURES / "tiny.marcxml", prepared, chunk_size=4)
    output_path = tmp_path / "results.jsonl"
    report = run_match(
        prepared_dir=prepared,
        expected_total=prepare_report.total_records,
        index_path=index_path,
        output_path=output_path,
        matching_config=config,
        pairing_config=pairing_config,
        idf=idf,
        author_idf=author_idf,
        publisher_idf=publisher_idf,
        workers=1,
        batch_size=2,
        queue_maxsize=4,
        report_interval_seconds=0.05,
        verbosity=1,
    )
    assert report.records_processed == prepare_report.total_records
    assert report.records_written == report.records_processed
    assert output_path.exists()


def _never_shutdown() -> bool:
    return False


def test_dead_workers_reports_only_exited() -> None:
    alive = _FakeProcess(name="w0", exitcode=None)
    dead = _FakeProcess(name="w1", exitcode=1)
    assert _dead_workers([alive, dead]) == ["w1"]


def test_dead_workers_empty_when_all_alive() -> None:
    procs = [_FakeProcess(name="w0"), _FakeProcess(name="w1")]
    assert _dead_workers(procs) == []


def test_guarded_put_succeeds_immediately() -> None:
    queue = _FakePutQueue(full_before_success=0)
    put = _make_guarded_put(queue, [], timeout=0.01, is_shutdown=_never_shutdown)
    put(b"blob")
    assert queue.puts == [b"blob"]


def test_guarded_put_retries_after_full_while_worker_alive() -> None:
    queue = _FakePutQueue(full_before_success=1)
    worker = _FakeProcess(name="w0", exitcode=None)
    put = _make_guarded_put(queue, [worker], timeout=0.01, is_shutdown=_never_shutdown)
    put(b"blob")
    assert queue.puts == [b"blob"]


def test_guarded_put_raises_when_worker_dead() -> None:
    queue = _FakePutQueue(full_before_success=1)
    worker = _FakeProcess(name="pd_matcher.worker.0", exitcode=1)
    put = _make_guarded_put(queue, [worker], timeout=0.01, is_shutdown=_never_shutdown)
    with raises(WorkerDiedError, match="exited unexpectedly") as excinfo:
        put(b"blob")
    assert "pd_matcher.worker.0" in str(excinfo.value)


def test_guarded_put_short_circuits_before_put_on_shutdown() -> None:
    queue = _FakePutQueue(full_before_success=0)
    put = _make_guarded_put(queue, [], timeout=0.01, is_shutdown=lambda: True)
    put(b"blob")
    assert queue.puts == []


def test_guarded_put_returns_on_shutdown_after_full() -> None:
    queue = _FakePutQueue(full_before_success=1)
    flags = iter([False, True])
    put = _make_guarded_put(
        queue, [_FakeProcess(name="w0")], timeout=0.01, is_shutdown=lambda: next(flags)
    )
    put(b"blob")
    assert queue.puts == []


def test_drain_sentinels_puts_one_none_per_worker() -> None:
    queue = _FakePutQueue(full_before_success=0)
    _drain_sentinels(queue, 3, timeout=0.01)
    assert queue.puts == [None, None, None]


def test_drain_sentinels_returns_early_on_full() -> None:
    queue = _FakePutQueue(full_before_success=1)
    _drain_sentinels(queue, 3, timeout=0.01)
    assert queue.puts == []


def test_run_match_worker_died_tears_down_and_reraises(
    pairing_config: PairingConfig,
    tmp_path: Path,
    monkeypatch: MonkeyPatch,
) -> None:
    """A ``WorkerDiedError`` from the producer drains the pool and propagates.

    The teardown must emit a ``worker_died`` shutdown event so the reporter
    thread stops (otherwise ``Reporter.__exit__`` would hang), terminate every
    child, and re-raise so the CLI exits nonzero.
    """
    index_path, idf, author_idf, publisher_idf = _build_tiny_index(tmp_path)

    def boom(*_args: object, **_kwargs: object) -> int:
        raise WorkerDiedError("worker(s) exited unexpectedly during production: w0")

    monkeypatch.setattr("pd_matcher.workers.pool.run_producer", boom)
    with raises(WorkerDiedError, match="exited unexpectedly"):
        run_match(
            marc_path=_FIXTURES / "tiny.marcxml",
            index_path=index_path,
            output_path=tmp_path / "results.jsonl",
            matching_config=_tiny_match_config(),
            pairing_config=pairing_config,
            idf=idf,
            author_idf=author_idf,
            publisher_idf=publisher_idf,
            workers=1,
            batch_size=2,
            queue_maxsize=4,
            report_interval_seconds=0.05,
        )
