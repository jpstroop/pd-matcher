"""Tests for :mod:`pd_matcher.workers.worker`."""

from collections.abc import Callable
from itertools import count
from itertools import cycle
from pathlib import Path

from _pytest.capture import CaptureFixture

from pd_matcher.config.schemas import CopyrightAssessmentConfig
from pd_matcher.config.schemas import CopyrightRuleSet
from pd_matcher.config.schemas import MatchingConfig
from pd_matcher.config.schemas import PairingConfig
from pd_matcher.index.lookup import NyplIndexLookup
from pd_matcher.logging_config import configure_logging
from pd_matcher.match.idf import IdfTable
from pd_matcher.match.pairing_compiler import CompiledPairings
from pd_matcher.models import MarcRecord
from pd_matcher.workers.events import RecordProcessed
from pd_matcher.workers.events import decode_stats_event
from pd_matcher.workers.messages import decode_worker_output
from pd_matcher.workers.producer import encode_batch
from pd_matcher.workers.worker import _WORKER_LOG_EVERY_N
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


def _drain_until(items: list[bytes], n: int) -> list[bytes]:
    return items[:n]


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


def test_worker_loop_processes_one_batch_and_stops_on_poison_pill(
    tiny_index_path: Path,
    tiny_idf: IdfTable,
    matching_config: MatchingConfig,
    copyright_config: CopyrightAssessmentConfig,
    ruleset: CopyrightRuleSet,
    compiled_pairings: CompiledPairings,
) -> None:
    outputs: list[bytes] = []
    stats: list[bytes] = []
    blobs: list[bytes | None] = [
        encode_batch((_make_marc(),)),
        None,
    ]
    with NyplIndexLookup(tiny_index_path) as lookup:
        processed = run_worker_loop(
            lookup=lookup,
            config=matching_config,
            idf=tiny_idf,
            calibrator=None,
            pairings=compiled_pairings,
            ruleset=ruleset,
            assessment_config=copyright_config,
            input_get=_build_input_get(blobs),
            output_put=outputs.append,
            stats_put=stats.append,
            is_shutdown=lambda: False,
        )
    assert processed == 1
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
    matching_config: MatchingConfig,
    copyright_config: CopyrightAssessmentConfig,
    ruleset: CopyrightRuleSet,
    compiled_pairings: CompiledPairings,
) -> None:
    outputs: list[bytes] = []
    stats: list[bytes] = []
    with NyplIndexLookup(tiny_index_path) as lookup:
        processed = run_worker_loop(
            lookup=lookup,
            config=matching_config,
            idf=tiny_idf,
            calibrator=None,
            pairings=compiled_pairings,
            ruleset=ruleset,
            assessment_config=copyright_config,
            input_get=_build_input_get([encode_batch((_make_marc(),))]),
            output_put=outputs.append,
            stats_put=stats.append,
            is_shutdown=lambda: True,
        )
    assert processed == 0
    assert outputs == []


def test_worker_loop_stops_between_records_when_shutdown_fires(
    tiny_index_path: Path,
    tiny_idf: IdfTable,
    matching_config: MatchingConfig,
    copyright_config: CopyrightAssessmentConfig,
    ruleset: CopyrightRuleSet,
    compiled_pairings: CompiledPairings,
) -> None:
    """A multi-record batch is aborted mid-batch when ``is_shutdown`` flips True."""
    outputs: list[bytes] = []
    stats: list[bytes] = []
    batch = encode_batch((_make_marc(), _make_marc(), _make_marc()))
    # is_shutdown returns False once (to enter loop, get blob), then True for
    # every record-level check that follows.
    flag = cycle([False, True])

    def is_shutdown() -> bool:
        return next(flag)

    with NyplIndexLookup(tiny_index_path) as lookup:
        processed = run_worker_loop(
            lookup=lookup,
            config=matching_config,
            idf=tiny_idf,
            calibrator=None,
            pairings=compiled_pairings,
            ruleset=ruleset,
            assessment_config=copyright_config,
            input_get=_build_input_get([batch, None]),
            output_put=outputs.append,
            stats_put=stats.append,
            is_shutdown=is_shutdown,
        )
    assert processed < 3


def test_worker_main_opens_lookup_and_runs_loop(
    tiny_index_path: Path,
    tiny_idf: IdfTable,
    matching_config: MatchingConfig,
    copyright_config: CopyrightAssessmentConfig,
    ruleset: CopyrightRuleSet,
    pairing_config: PairingConfig,
) -> None:
    outputs: list[bytes] = []
    stats: list[bytes] = []
    blobs: list[bytes | None] = [encode_batch((_make_marc(),)), None]
    processed = worker_main(
        index_path=tiny_index_path,
        matching_config=matching_config,
        copyright_config=copyright_config,
        ruleset=ruleset,
        pairing_config=pairing_config,
        idf=tiny_idf,
        calibrator=None,
        input_get=_build_input_get(blobs),
        output_put=outputs.append,
        stats_put=stats.append,
        is_shutdown=lambda: False,
    )
    assert processed == 1


def test_worker_main_with_unmatchable_record_emits_blank_match(
    tiny_index_path: Path,
    tiny_idf: IdfTable,
    matching_config: MatchingConfig,
    copyright_config: CopyrightAssessmentConfig,
    ruleset: CopyrightRuleSet,
    pairing_config: PairingConfig,
) -> None:
    outputs: list[bytes] = []
    stats: list[bytes] = []
    marc = MarcRecord(control_id="orphan", title="nothing relevant", title_main="nothing relevant")
    blobs: list[bytes | None] = [encode_batch((marc,)), None]
    processed = worker_main(
        index_path=tiny_index_path,
        matching_config=matching_config,
        copyright_config=copyright_config,
        ruleset=ruleset,
        pairing_config=pairing_config,
        idf=tiny_idf,
        calibrator=None,
        input_get=_build_input_get(blobs),
        output_put=outputs.append,
        stats_put=stats.append,
        is_shutdown=lambda: False,
    )
    assert processed == 1
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
    matching_config: MatchingConfig,
    copyright_config: CopyrightAssessmentConfig,
    ruleset: CopyrightRuleSet,
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
            calibrator=None,
            pairings=compiled_pairings,
            ruleset=ruleset,
            assessment_config=copyright_config,
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
    matching_config: MatchingConfig,
    copyright_config: CopyrightAssessmentConfig,
    ruleset: CopyrightRuleSet,
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
            calibrator=None,
            pairings=compiled_pairings,
            ruleset=ruleset,
            assessment_config=copyright_config,
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
    matching_config: MatchingConfig,
    copyright_config: CopyrightAssessmentConfig,
    ruleset: CopyrightRuleSet,
    compiled_pairings: CompiledPairings,
    capsys: CaptureFixture[str],
) -> None:
    """``-vv`` logs one hit line per record, carrying marc id and status."""
    configure_logging(level="INFO", json_output=False)
    blobs: list[bytes | None] = [encode_batch((_make_marc(),)), None]
    with NyplIndexLookup(tiny_index_path) as lookup:
        run_worker_loop(
            lookup=lookup,
            config=matching_config,
            idf=tiny_idf,
            calibrator=None,
            pairings=compiled_pairings,
            ruleset=ruleset,
            assessment_config=copyright_config,
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
    assert "status=" in err


def test_worker_loop_logs_progress_every_n_records(
    tiny_index_path: Path,
    tiny_idf: IdfTable,
    matching_config: MatchingConfig,
    copyright_config: CopyrightAssessmentConfig,
    ruleset: CopyrightRuleSet,
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
            calibrator=None,
            pairings=compiled_pairings,
            ruleset=ruleset,
            assessment_config=copyright_config,
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
    matching_config: MatchingConfig,
    copyright_config: CopyrightAssessmentConfig,
    ruleset: CopyrightRuleSet,
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
            calibrator=None,
            pairings=compiled_pairings,
            ruleset=ruleset,
            assessment_config=copyright_config,
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
    matching_config: MatchingConfig,
    copyright_config: CopyrightAssessmentConfig,
    ruleset: CopyrightRuleSet,
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
            calibrator=None,
            pairings=compiled_pairings,
            ruleset=ruleset,
            assessment_config=copyright_config,
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
    matching_config: MatchingConfig,
    copyright_config: CopyrightAssessmentConfig,
    ruleset: CopyrightRuleSet,
    pairing_config: PairingConfig,
    capsys: CaptureFixture[str],
) -> None:
    """``worker_main`` forwards ``worker_id``/``verbosity`` into the loop's logs."""
    configure_logging(level="INFO", json_output=False)
    blobs: list[bytes | None] = [encode_batch((_make_marc(),)), None]
    processed = worker_main(
        index_path=tiny_index_path,
        matching_config=matching_config,
        copyright_config=copyright_config,
        ruleset=ruleset,
        pairing_config=pairing_config,
        idf=tiny_idf,
        calibrator=None,
        input_get=_build_input_get(blobs),
        output_put=_sink(),
        stats_put=_sink(),
        is_shutdown=lambda: False,
        worker_id=9,
        verbosity=1,
    )
    assert processed == 1
    assert "worker=9" in capsys.readouterr().err


def test_worker_main_uses_default_year_when_assessment_config_unpinned(
    tiny_index_path: Path,
    tiny_idf: IdfTable,
    matching_config: MatchingConfig,
    ruleset: CopyrightRuleSet,
    pairing_config: PairingConfig,
) -> None:
    """When ``copyright_config.as_of_year is None`` the worker uses the current year."""
    outputs: list[bytes] = []
    stats: list[bytes] = []
    blobs: list[bytes | None] = [encode_batch((_make_marc(),)), None]
    processed = worker_main(
        index_path=tiny_index_path,
        matching_config=matching_config,
        copyright_config=CopyrightAssessmentConfig(as_of_year=None),
        ruleset=ruleset,
        pairing_config=pairing_config,
        idf=tiny_idf,
        calibrator=None,
        input_get=_build_input_get(blobs),
        output_put=outputs.append,
        stats_put=stats.append,
        is_shutdown=lambda: False,
    )
    assert processed == 1
