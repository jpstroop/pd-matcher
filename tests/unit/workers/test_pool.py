"""Tests for :mod:`pd_matcher.workers.pool` (non-spawn paths)."""

from csv import DictReader
from datetime import date
from multiprocessing import Event
from multiprocessing import get_context
from multiprocessing.queues import Queue as MpQueue
from pathlib import Path

from pytest import raises

from pd_matcher.config.loader import load_copyright_rules
from pd_matcher.config.schemas import CopyrightAssessmentConfig
from pd_matcher.config.schemas import CopyrightRuleSet
from pd_matcher.config.schemas import MatchingConfig
from pd_matcher.copyright.assessment import CopyrightAssessment
from pd_matcher.copyright.status import CopyrightStatus
from pd_matcher.index.lookup import NyplIndexLookup
from pd_matcher.match.idf import IdfTable
from pd_matcher.match.idf import build_idf_table
from pd_matcher.match.result import MatchResult
from pd_matcher.models import MarcRecord
from pd_matcher.output.csv_writer import CsvResultWriter
from pd_matcher.workers.events import decode_stats_event
from pd_matcher.workers.messages import WorkerOutput
from pd_matcher.workers.messages import encode_worker_output
from pd_matcher.workers.pool import RunReport
from pd_matcher.workers.pool import _build_csv_writer
from pd_matcher.workers.pool import _default_workers
from pd_matcher.workers.pool import _shutdown_predicate
from pd_matcher.workers.pool import _worker_entry
from pd_matcher.workers.pool import _writer_entry
from pd_matcher.workers.pool import run_match
from pd_matcher.workers.producer import encode_batch

_DEFAULTS = (
    Path(__file__).resolve().parents[3]
    / "src"
    / "pd_matcher"
    / "config"
    / "defaults"
    / "copyright_rules.yaml"
)
_FIXTURES = Path(__file__).resolve().parents[2] / "fixtures"


def test_default_workers_returns_positive_count() -> None:
    assert _default_workers() >= 1


def test_build_csv_writer_returns_csv_writer(tmp_path: Path) -> None:
    writer = _build_csv_writer(tmp_path / "x.csv")
    assert isinstance(writer, CsvResultWriter)


def test_shutdown_predicate_tracks_event() -> None:
    event = Event()
    predicate = _shutdown_predicate(event)
    assert predicate() is False
    event.set()
    assert predicate() is True


def test_worker_entry_runs_in_process_against_real_queues(
    tiny_index_path: Path,
    tiny_idf: IdfTable,
    matching_config: MatchingConfig,
    copyright_config: CopyrightAssessmentConfig,
    ruleset: CopyrightRuleSet,
) -> None:
    """``_worker_entry`` is reachable in-process via real spawn-context queues."""
    ctx = get_context("spawn")
    input_queue: MpQueue[bytes | None] = ctx.Queue()
    output_queue: MpQueue[bytes | None] = ctx.Queue()
    stats_queue: MpQueue[bytes] = ctx.Queue()
    event = ctx.Event()
    marc = MarcRecord(control_id="m", title="t", publication_year=1940)
    input_queue.put(encode_batch((marc,)))
    input_queue.put(None)
    _worker_entry(
        index_path=tiny_index_path,
        matching_config=matching_config,
        copyright_config=copyright_config,
        ruleset=ruleset,
        idf=tiny_idf,
        calibrator=None,
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
    marc = MarcRecord(control_id="m", title="t")
    payload = WorkerOutput(
        marc=marc,
        match=MatchResult(
            marc_control_id="m",
            best=None,
            alternates=(),
            candidates_considered=0,
        ),
        assessment=CopyrightAssessment(
            status=CopyrightStatus.UNKNOWN_INSUFFICIENT_DATA,
            matched_rule_name=None,
            explanation="",
            assumptions=(),
        ),
        matched_nypl=None,
    )
    output_queue.put(encode_worker_output(payload))
    output_queue.put(None)
    path = tmp_path / "out.csv"
    _writer_entry(
        output_path=path,
        writer_factory=CsvResultWriter,
        output_queue=output_queue,
        stats_queue=stats_queue,
        shutdown_event=event,
    )
    rows = list(DictReader(path.open(encoding="utf-8")))
    assert len(rows) == 1


def test_run_match_rejects_zero_workers(
    tiny_index_path: Path,
    tiny_idf: IdfTable,
    matching_config: MatchingConfig,
    copyright_config: CopyrightAssessmentConfig,
    tmp_path: Path,
) -> None:
    ruleset = load_copyright_rules(_DEFAULTS)
    with raises(ValueError, match="workers must be >= 1"):
        run_match(
            marc_path=_FIXTURES / "tiny.marcxml",
            index_path=tiny_index_path,
            output_path=tmp_path / "results.csv",
            matching_config=matching_config,
            copyright_config=copyright_config,
            ruleset=ruleset,
            idf=tiny_idf,
            workers=0,
        )


def test_run_match_returns_run_report(
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
    config = MatchingConfig(
        title_weight=0.40,
        author_weight=0.20,
        publisher_weight=0.10,
        year_weight=0.10,
        edition_weight=0.05,
        lccn_weight=0.10,
        isbn_weight=0.05,
        year_window=2,
        min_combined_score=30.0,
        scorer="weighted_mean",
    )
    copyright_config = CopyrightAssessmentConfig(today=date(2026, 5, 18))
    ruleset = load_copyright_rules(_DEFAULTS)
    output_path = tmp_path / "results.csv"
    report = run_match(
        marc_path=_FIXTURES / "tiny.marcxml",
        index_path=index_path,
        output_path=output_path,
        matching_config=config,
        copyright_config=copyright_config,
        ruleset=ruleset,
        idf=idf,
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
