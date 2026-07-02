"""Tests for :mod:`pd_matcher.workers.worker`."""

from collections.abc import Callable
from itertools import count
from itertools import cycle
from pathlib import Path

from _pytest.capture import CaptureFixture
from _pytest.monkeypatch import MonkeyPatch
from lmdb import Error as LmdbError
from pytest import raises

from pd_matcher.config.schemas import MatchingConfig
from pd_matcher.config.schemas import PairingConfig
from pd_matcher.index.lookup import NyplIndexLookup
from pd_matcher.logging_config import configure_logging
from pd_matcher.match.combiners.base import Combiner
from pd_matcher.match.combiners.calibrator import PlattCalibrator
from pd_matcher.match.idf import IdfTable
from pd_matcher.match.pairing_compiler import CompiledPairings
from pd_matcher.models import MarcRecord
from pd_matcher.workers.events import RecordProcessed
from pd_matcher.workers.events import RecordSkipped
from pd_matcher.workers.events import decode_stats_event
from pd_matcher.workers.messages import WorkerOutput
from pd_matcher.workers.messages import decode_worker_output
from pd_matcher.workers.producer import encode_batch
from pd_matcher.workers.worker import _WORKER_LOG_EVERY_N
from pd_matcher.workers.worker import _process_record
from pd_matcher.workers.worker import _worker_rate
from pd_matcher.workers.worker import run_worker_loop
from pd_matcher.workers.worker import worker_main


def _make_marc() -> MarcRecord:
    return MarcRecord(
        control_id="m-1",
        title="A study of widgets",
        title_main="A study of widgets",
        main_author="Smith, John",
        publisher="Acme Press",
        edition="1st ed.",
        publication_year=1940,
        country_code="nyu",
        language_code="eng",
    )


def _build_input_get(blobs: list[bytes | None]) -> Callable[[], bytes | None]:
    """Wrap a static list of pre-encoded batches as a queue.get-style callable."""
    iterator = iter(blobs)

    def get() -> bytes | None:
        return next(iterator)

    return get


def _sink() -> Callable[[bytes], None]:
    """Return a put-style callable that discards the bytes it receives."""

    def put(_blob: bytes) -> None:
        return None

    return put


def test_worker_loop_runs_under_thread_limit_guard(
    tiny_index_path: Path,
    tiny_idf: IdfTable,
    tiny_author_idf: IdfTable,
    tiny_publisher_idf: IdfTable,
    matching_config: MatchingConfig,
    compiled_pairings: CompiledPairings,
    monkeypatch: MonkeyPatch,
) -> None:
    """The consume loop is wrapped in the runtime thread-limit guard (#101)."""
    from collections.abc import Iterator
    from contextlib import contextmanager

    entered: list[bool] = []

    @contextmanager
    def spy() -> Iterator[None]:
        entered.append(True)
        yield

    monkeypatch.setattr("pd_matcher.workers.worker.limit_worker_threads", spy)
    outputs: list[bytes] = []
    stats: list[bytes] = []
    blobs: list[bytes | None] = [encode_batch((_make_marc(),)), None]
    with NyplIndexLookup(tiny_index_path) as lookup:
        run_worker_loop(
            lookup=lookup,
            config=matching_config,
            idf=tiny_idf,
            author_idf=tiny_author_idf,
            publisher_idf=tiny_publisher_idf,
            calibrator=None,
            learned_model_dir=None,
            pairings=compiled_pairings,
            input_get=_build_input_get(blobs),
            output_put=outputs.append,
            stats_put=stats.append,
            is_shutdown=lambda: False,
        )
    assert entered == [True]


def test_worker_loop_processes_one_batch_and_stops_on_poison_pill(
    tiny_index_path: Path,
    tiny_idf: IdfTable,
    tiny_author_idf: IdfTable,
    tiny_publisher_idf: IdfTable,
    matching_config: MatchingConfig,
    compiled_pairings: CompiledPairings,
) -> None:
    outputs: list[bytes] = []
    stats: list[bytes] = []
    blobs: list[bytes | None] = [
        encode_batch((_make_marc(),)),
        None,
    ]
    with NyplIndexLookup(tiny_index_path) as lookup:
        result = run_worker_loop(
            lookup=lookup,
            config=matching_config,
            idf=tiny_idf,
            author_idf=tiny_author_idf,
            publisher_idf=tiny_publisher_idf,
            calibrator=None,
            learned_model_dir=None,
            pairings=compiled_pairings,
            input_get=_build_input_get(blobs),
            output_put=outputs.append,
            stats_put=stats.append,
            is_shutdown=lambda: False,
        )
    assert result.processed == 1
    assert result.skipped == 0
    assert len(outputs) == 1
    payload = decode_worker_output(outputs[0])
    assert payload.marc.control_id == "m-1"
    assert payload.match is not None
    assert payload.match.best is not None
    assert payload.matched_nypl is not None
    event = decode_stats_event(stats[0])
    assert isinstance(event, RecordProcessed)
    assert event.candidates_considered >= 1
    assert 0.0 <= event.confidence <= 1.0


def test_worker_loop_stops_when_shutdown_before_processing(
    tiny_index_path: Path,
    tiny_idf: IdfTable,
    tiny_author_idf: IdfTable,
    tiny_publisher_idf: IdfTable,
    matching_config: MatchingConfig,
    compiled_pairings: CompiledPairings,
) -> None:
    outputs: list[bytes] = []
    stats: list[bytes] = []
    with NyplIndexLookup(tiny_index_path) as lookup:
        result = run_worker_loop(
            lookup=lookup,
            config=matching_config,
            idf=tiny_idf,
            author_idf=tiny_author_idf,
            publisher_idf=tiny_publisher_idf,
            calibrator=None,
            learned_model_dir=None,
            pairings=compiled_pairings,
            input_get=_build_input_get([encode_batch((_make_marc(),))]),
            output_put=outputs.append,
            stats_put=stats.append,
            is_shutdown=lambda: True,
        )
    assert result.processed == 0
    assert outputs == []


def test_worker_loop_stops_between_records_when_shutdown_fires(
    tiny_index_path: Path,
    tiny_idf: IdfTable,
    tiny_author_idf: IdfTable,
    tiny_publisher_idf: IdfTable,
    matching_config: MatchingConfig,
    compiled_pairings: CompiledPairings,
) -> None:
    """A multi-record batch is aborted mid-batch when ``is_shutdown`` flips True."""
    outputs: list[bytes] = []
    stats: list[bytes] = []
    batch = encode_batch((_make_marc(), _make_marc(), _make_marc()))
    flag = cycle([False, True])

    def is_shutdown() -> bool:
        return next(flag)

    with NyplIndexLookup(tiny_index_path) as lookup:
        result = run_worker_loop(
            lookup=lookup,
            config=matching_config,
            idf=tiny_idf,
            author_idf=tiny_author_idf,
            publisher_idf=tiny_publisher_idf,
            calibrator=None,
            learned_model_dir=None,
            pairings=compiled_pairings,
            input_get=_build_input_get([batch, None]),
            output_put=outputs.append,
            stats_put=stats.append,
            is_shutdown=is_shutdown,
        )
    assert result.processed < 3


def test_worker_main_opens_lookup_and_runs_loop(
    tiny_index_path: Path,
    tiny_idf: IdfTable,
    tiny_author_idf: IdfTable,
    tiny_publisher_idf: IdfTable,
    matching_config: MatchingConfig,
    pairing_config: PairingConfig,
) -> None:
    outputs: list[bytes] = []
    stats: list[bytes] = []
    blobs: list[bytes | None] = [encode_batch((_make_marc(),)), None]
    result = worker_main(
        index_path=tiny_index_path,
        matching_config=matching_config,
        pairing_config=pairing_config,
        idf=tiny_idf,
        author_idf=tiny_author_idf,
        publisher_idf=tiny_publisher_idf,
        calibrator=None,
        learned_model_dir=None,
        input_get=_build_input_get(blobs),
        output_put=outputs.append,
        stats_put=stats.append,
        is_shutdown=lambda: False,
    )
    assert result.processed == 1


def test_worker_main_with_unmatchable_record_emits_blank_match(
    tiny_index_path: Path,
    tiny_idf: IdfTable,
    tiny_author_idf: IdfTable,
    tiny_publisher_idf: IdfTable,
    matching_config: MatchingConfig,
    pairing_config: PairingConfig,
) -> None:
    outputs: list[bytes] = []
    stats: list[bytes] = []
    marc = MarcRecord(control_id="orphan", title="nothing relevant", title_main="nothing relevant")
    blobs: list[bytes | None] = [encode_batch((marc,)), None]
    result = worker_main(
        index_path=tiny_index_path,
        matching_config=matching_config,
        pairing_config=pairing_config,
        idf=tiny_idf,
        author_idf=tiny_author_idf,
        publisher_idf=tiny_publisher_idf,
        calibrator=None,
        learned_model_dir=None,
        input_get=_build_input_get(blobs),
        output_put=outputs.append,
        stats_put=stats.append,
        is_shutdown=lambda: False,
    )
    assert result.processed == 1
    payload = decode_worker_output(outputs[0])
    assert payload.match is not None
    assert payload.match.best is None
    assert payload.matched_nypl is None
    event = decode_stats_event(stats[0])
    assert isinstance(event, RecordProcessed)
    assert event.candidates_considered == 0


def test_worker_rate_returns_zero_when_no_elapsed_time() -> None:
    """A worker that has done nothing in zero wall time has rate ``0.0``."""
    assert _worker_rate(processed=0, started_at=5.0, now=5.0) == 0.0


def test_worker_rate_divides_processed_by_elapsed() -> None:
    assert _worker_rate(processed=100, started_at=0.0, now=10.0) == 10.0


def test_worker_loop_silent_at_verbosity_zero(
    tiny_index_path: Path,
    tiny_idf: IdfTable,
    tiny_author_idf: IdfTable,
    tiny_publisher_idf: IdfTable,
    matching_config: MatchingConfig,
    compiled_pairings: CompiledPairings,
    capsys: CaptureFixture[str],
) -> None:
    """Default verbosity emits no per-worker lines."""
    configure_logging(level="INFO", json_output=False)
    blobs: list[bytes | None] = [encode_batch((_make_marc(),)), None]
    with NyplIndexLookup(tiny_index_path) as lookup:
        run_worker_loop(
            lookup=lookup,
            config=matching_config,
            idf=tiny_idf,
            author_idf=tiny_author_idf,
            publisher_idf=tiny_publisher_idf,
            calibrator=None,
            learned_model_dir=None,
            pairings=compiled_pairings,
            input_get=_build_input_get(blobs),
            output_put=_sink(),
            stats_put=_sink(),
            is_shutdown=lambda: False,
            worker_id=3,
            verbosity=0,
        )
    err = capsys.readouterr().err
    assert "worker.start" not in err
    assert "worker.finish" not in err


def test_worker_loop_logs_start_and_finish_with_id_and_rate_at_v(
    tiny_index_path: Path,
    tiny_idf: IdfTable,
    tiny_author_idf: IdfTable,
    tiny_publisher_idf: IdfTable,
    matching_config: MatchingConfig,
    compiled_pairings: CompiledPairings,
    capsys: CaptureFixture[str],
) -> None:
    """``-v`` logs worker id + rate on start and finish; no per-record hits."""
    configure_logging(level="INFO", json_output=False)
    ticks = count(0, 1)
    blobs: list[bytes | None] = [encode_batch((_make_marc(),)), None]
    with NyplIndexLookup(tiny_index_path) as lookup:
        run_worker_loop(
            lookup=lookup,
            config=matching_config,
            idf=tiny_idf,
            author_idf=tiny_author_idf,
            publisher_idf=tiny_publisher_idf,
            calibrator=None,
            learned_model_dir=None,
            pairings=compiled_pairings,
            input_get=_build_input_get(blobs),
            output_put=_sink(),
            stats_put=_sink(),
            is_shutdown=lambda: False,
            worker_id=7,
            verbosity=1,
            clock=lambda: float(next(ticks)),
        )
    err = capsys.readouterr().err
    assert "worker.start" in err
    assert "worker.finish" in err
    assert "worker=7" in err
    assert "rate=" in err
    assert "worker.hit" not in err


def test_worker_loop_logs_per_record_hits_at_vv(
    tiny_index_path: Path,
    tiny_idf: IdfTable,
    tiny_author_idf: IdfTable,
    tiny_publisher_idf: IdfTable,
    matching_config: MatchingConfig,
    compiled_pairings: CompiledPairings,
    capsys: CaptureFixture[str],
) -> None:
    """``-vv`` logs one hit line per record, carrying marc id."""
    configure_logging(level="INFO", json_output=False)
    blobs: list[bytes | None] = [encode_batch((_make_marc(),)), None]
    with NyplIndexLookup(tiny_index_path) as lookup:
        run_worker_loop(
            lookup=lookup,
            config=matching_config,
            idf=tiny_idf,
            author_idf=tiny_author_idf,
            publisher_idf=tiny_publisher_idf,
            calibrator=None,
            learned_model_dir=None,
            pairings=compiled_pairings,
            input_get=_build_input_get(blobs),
            output_put=_sink(),
            stats_put=_sink(),
            is_shutdown=lambda: False,
            worker_id=2,
            verbosity=2,
        )
    err = capsys.readouterr().err
    assert "worker.hit" in err
    assert "marc=m-1" in err


def test_worker_loop_logs_progress_every_n_records(
    tiny_index_path: Path,
    tiny_idf: IdfTable,
    tiny_author_idf: IdfTable,
    tiny_publisher_idf: IdfTable,
    matching_config: MatchingConfig,
    compiled_pairings: CompiledPairings,
    capsys: CaptureFixture[str],
) -> None:
    """A worker crossing the every-N threshold emits an interim progress line."""
    configure_logging(level="INFO", json_output=False)
    records = tuple(_make_marc() for _ in range(_WORKER_LOG_EVERY_N))
    blobs: list[bytes | None] = [encode_batch(records), None]
    with NyplIndexLookup(tiny_index_path) as lookup:
        run_worker_loop(
            lookup=lookup,
            config=matching_config,
            idf=tiny_idf,
            author_idf=tiny_author_idf,
            publisher_idf=tiny_publisher_idf,
            calibrator=None,
            learned_model_dir=None,
            pairings=compiled_pairings,
            input_get=_build_input_get(blobs),
            output_put=_sink(),
            stats_put=_sink(),
            is_shutdown=lambda: False,
            worker_id=1,
            verbosity=1,
        )
    err = capsys.readouterr().err
    assert "worker.progress" in err


def test_worker_loop_finish_logged_when_shutdown_mid_batch_at_v(
    tiny_index_path: Path,
    tiny_idf: IdfTable,
    tiny_author_idf: IdfTable,
    tiny_publisher_idf: IdfTable,
    matching_config: MatchingConfig,
    compiled_pairings: CompiledPairings,
    capsys: CaptureFixture[str],
) -> None:
    """The mid-batch shutdown exit still logs a finish line under ``-v``."""
    configure_logging(level="INFO", json_output=False)
    batch = encode_batch((_make_marc(), _make_marc()))
    flag = cycle([False, True])
    with NyplIndexLookup(tiny_index_path) as lookup:
        run_worker_loop(
            lookup=lookup,
            config=matching_config,
            idf=tiny_idf,
            author_idf=tiny_author_idf,
            publisher_idf=tiny_publisher_idf,
            calibrator=None,
            learned_model_dir=None,
            pairings=compiled_pairings,
            input_get=_build_input_get([batch, None]),
            output_put=_sink(),
            stats_put=_sink(),
            is_shutdown=lambda: next(flag),
            worker_id=4,
            verbosity=1,
        )
    err = capsys.readouterr().err
    assert "worker.finish" in err
    assert "worker=4" in err


def test_worker_loop_hit_line_marks_no_match_at_vv(
    tiny_index_path: Path,
    tiny_idf: IdfTable,
    tiny_author_idf: IdfTable,
    tiny_publisher_idf: IdfTable,
    matching_config: MatchingConfig,
    compiled_pairings: CompiledPairings,
    capsys: CaptureFixture[str],
) -> None:
    """An unmatchable record logs ``reg=none`` and ``score=0.0`` under ``-vv``."""
    configure_logging(level="INFO", json_output=False)
    orphan = MarcRecord(
        control_id="orphan", title="nothing relevant", title_main="nothing relevant"
    )
    blobs: list[bytes | None] = [encode_batch((orphan,)), None]
    with NyplIndexLookup(tiny_index_path) as lookup:
        run_worker_loop(
            lookup=lookup,
            config=matching_config,
            idf=tiny_idf,
            author_idf=tiny_author_idf,
            publisher_idf=tiny_publisher_idf,
            calibrator=None,
            learned_model_dir=None,
            pairings=compiled_pairings,
            input_get=_build_input_get(blobs),
            output_put=_sink(),
            stats_put=_sink(),
            is_shutdown=lambda: False,
            worker_id=0,
            verbosity=2,
        )
    err = capsys.readouterr().err
    assert "reg=none" in err
    assert "marc=orphan" in err


def test_worker_main_threads_worker_id_and_verbosity(
    tiny_index_path: Path,
    tiny_idf: IdfTable,
    tiny_author_idf: IdfTable,
    tiny_publisher_idf: IdfTable,
    matching_config: MatchingConfig,
    pairing_config: PairingConfig,
    capsys: CaptureFixture[str],
) -> None:
    """``worker_main`` forwards ``worker_id``/``verbosity`` into the loop's logs."""
    configure_logging(level="INFO", json_output=False)
    blobs: list[bytes | None] = [encode_batch((_make_marc(),)), None]
    result = worker_main(
        index_path=tiny_index_path,
        matching_config=matching_config,
        pairing_config=pairing_config,
        idf=tiny_idf,
        author_idf=tiny_author_idf,
        publisher_idf=tiny_publisher_idf,
        calibrator=None,
        learned_model_dir=None,
        input_get=_build_input_get(blobs),
        output_put=_sink(),
        stats_put=_sink(),
        is_shutdown=lambda: False,
        worker_id=9,
        verbosity=1,
    )
    assert result.processed == 1
    assert "worker=9" in capsys.readouterr().err


def _process_record_raising_on_boom(
    marc: MarcRecord,
    *,
    lookup: NyplIndexLookup,
    config: MatchingConfig,
    idf: IdfTable,
    author_idf: IdfTable,
    publisher_idf: IdfTable,
    calibrator: PlattCalibrator | None,
    combiner: Combiner,
    pairings: CompiledPairings,
) -> WorkerOutput:
    """Stand-in for ``_process_record`` that raises only on the ``boom`` record."""
    if marc.control_id == "boom":
        raise RuntimeError("synthetic record failure")
    return _process_record(
        marc,
        lookup=lookup,
        config=config,
        idf=idf,
        author_idf=author_idf,
        publisher_idf=publisher_idf,
        calibrator=calibrator,
        combiner=combiner,
        pairings=pairings,
    )


def test_worker_loop_skips_failing_record_and_continues(
    tiny_index_path: Path,
    tiny_idf: IdfTable,
    tiny_author_idf: IdfTable,
    tiny_publisher_idf: IdfTable,
    matching_config: MatchingConfig,
    compiled_pairings: CompiledPairings,
    monkeypatch: MonkeyPatch,
) -> None:
    """A record whose processing raises is skipped; later records still run."""
    monkeypatch.setattr(
        "pd_matcher.workers.worker._process_record", _process_record_raising_on_boom
    )
    good_before = _make_marc()
    bad = MarcRecord(control_id="boom", title="kaboom", title_main="kaboom")
    good_after = MarcRecord(
        control_id="m-after", title="A study of widgets", title_main="A study of widgets"
    )
    outputs: list[bytes] = []
    stats: list[bytes] = []
    blobs: list[bytes | None] = [encode_batch((good_before, bad, good_after)), None]
    with NyplIndexLookup(tiny_index_path) as lookup:
        result = run_worker_loop(
            lookup=lookup,
            config=matching_config,
            idf=tiny_idf,
            author_idf=tiny_author_idf,
            publisher_idf=tiny_publisher_idf,
            calibrator=None,
            learned_model_dir=None,
            pairings=compiled_pairings,
            input_get=_build_input_get(blobs),
            output_put=outputs.append,
            stats_put=stats.append,
            is_shutdown=lambda: False,
        )
    assert result.processed == 2
    assert result.skipped == 1
    processed_ids = {decode_worker_output(blob).marc.control_id for blob in outputs}
    assert processed_ids == {"m-1", "m-after"}
    skip_events = [
        event
        for event in (decode_stats_event(blob) for blob in stats)
        if isinstance(event, RecordSkipped)
    ]
    assert len(skip_events) == 1
    assert skip_events[0].control_id == "boom"


def test_worker_main_logs_and_reraises_on_fatal_crash(
    tmp_path: Path,
    tiny_idf: IdfTable,
    tiny_author_idf: IdfTable,
    tiny_publisher_idf: IdfTable,
    matching_config: MatchingConfig,
    pairing_config: PairingConfig,
    capsys: CaptureFixture[str],
) -> None:
    """A fatal crash (bad index path) logs ``worker.crashed`` and re-raises."""
    configure_logging(level="INFO", json_output=False)
    with raises(LmdbError):
        worker_main(
            index_path=tmp_path / "does-not-exist.lmdb",
            matching_config=matching_config,
            pairing_config=pairing_config,
            idf=tiny_idf,
            author_idf=tiny_author_idf,
            publisher_idf=tiny_publisher_idf,
            calibrator=None,
            learned_model_dir=None,
            input_get=_build_input_get([None]),
            output_put=_sink(),
            stats_put=_sink(),
            is_shutdown=lambda: False,
            worker_id=7,
        )
    err = capsys.readouterr().err
    assert "worker.crashed" in err
    assert "worker=7" in err


def test_worker_loop_logs_failing_record_with_worker_and_marc_id(
    tiny_index_path: Path,
    tiny_idf: IdfTable,
    tiny_author_idf: IdfTable,
    tiny_publisher_idf: IdfTable,
    matching_config: MatchingConfig,
    compiled_pairings: CompiledPairings,
    monkeypatch: MonkeyPatch,
    capsys: CaptureFixture[str],
) -> None:
    """A failing record is logged at error level with worker id, marc id, traceback."""
    configure_logging(level="INFO", json_output=False)
    monkeypatch.setattr(
        "pd_matcher.workers.worker._process_record", _process_record_raising_on_boom
    )
    bad = MarcRecord(control_id="boom", title="kaboom", title_main="kaboom")
    blobs: list[bytes | None] = [encode_batch((bad,)), None]
    with NyplIndexLookup(tiny_index_path) as lookup:
        result = run_worker_loop(
            lookup=lookup,
            config=matching_config,
            idf=tiny_idf,
            author_idf=tiny_author_idf,
            publisher_idf=tiny_publisher_idf,
            calibrator=None,
            learned_model_dir=None,
            pairings=compiled_pairings,
            input_get=_build_input_get(blobs),
            output_put=_sink(),
            stats_put=_sink(),
            is_shutdown=lambda: False,
            worker_id=5,
        )
    assert result.processed == 0
    assert result.skipped == 1
    err = capsys.readouterr().err
    assert "worker.record_failed" in err
    assert "worker=5" in err
    assert "marc=boom" in err
    assert "synthetic record failure" in err
