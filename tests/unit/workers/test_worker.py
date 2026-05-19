"""Tests for :mod:`pd_matcher.workers.worker`."""

from collections.abc import Callable
from itertools import cycle
from pathlib import Path

from pd_matcher.config.schemas import CopyrightAssessmentConfig
from pd_matcher.config.schemas import CopyrightRuleSet
from pd_matcher.config.schemas import MatchingConfig
from pd_matcher.index.lookup import NyplIndexLookup
from pd_matcher.match.idf import IdfTable
from pd_matcher.models import MarcRecord
from pd_matcher.workers.events import RecordProcessed
from pd_matcher.workers.events import decode_stats_event
from pd_matcher.workers.messages import decode_worker_output
from pd_matcher.workers.producer import encode_batch
from pd_matcher.workers.worker import run_worker_loop
from pd_matcher.workers.worker import worker_main


def _make_marc() -> MarcRecord:
    return MarcRecord(
        control_id="m-1",
        title="A study of widgets",
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


def test_worker_loop_processes_one_batch_and_stops_on_poison_pill(
    tiny_index_path: Path,
    tiny_idf: IdfTable,
    matching_config: MatchingConfig,
    copyright_config: CopyrightAssessmentConfig,
    ruleset: CopyrightRuleSet,
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
) -> None:
    outputs: list[bytes] = []
    stats: list[bytes] = []
    with NyplIndexLookup(tiny_index_path) as lookup:
        processed = run_worker_loop(
            lookup=lookup,
            config=matching_config,
            idf=tiny_idf,
            calibrator=None,
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
) -> None:
    outputs: list[bytes] = []
    stats: list[bytes] = []
    blobs: list[bytes | None] = [encode_batch((_make_marc(),)), None]
    processed = worker_main(
        index_path=tiny_index_path,
        matching_config=matching_config,
        copyright_config=copyright_config,
        ruleset=ruleset,
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
) -> None:
    outputs: list[bytes] = []
    stats: list[bytes] = []
    marc = MarcRecord(control_id="orphan", title="nothing relevant")
    blobs: list[bytes | None] = [encode_batch((marc,)), None]
    processed = worker_main(
        index_path=tiny_index_path,
        matching_config=matching_config,
        copyright_config=copyright_config,
        ruleset=ruleset,
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


def test_worker_main_uses_default_year_when_assessment_config_unpinned(
    tiny_index_path: Path,
    tiny_idf: IdfTable,
    matching_config: MatchingConfig,
    ruleset: CopyrightRuleSet,
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
        idf=tiny_idf,
        calibrator=None,
        input_get=_build_input_get(blobs),
        output_put=outputs.append,
        stats_put=stats.append,
        is_shutdown=lambda: False,
    )
    assert processed == 1
