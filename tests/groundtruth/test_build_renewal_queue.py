"""Unit tests for the renewal-first review-queue builder."""

from __future__ import annotations

from collections.abc import Iterator
from collections.abc import Sequence
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from typing import cast

from pytest import LogCaptureFixture
from pytest import MonkeyPatch
from pytest import raises

from pd_groundtruth import build_renewal_queue as module
from pd_groundtruth.build_renewal_queue import RenewalScore
from pd_groundtruth.build_renewal_queue import best_renewal
from pd_groundtruth.build_renewal_queue import build_renewal_queue
from pd_groundtruth.build_renewal_queue import score_renewal
from pd_groundtruth.review_db import PAIRING_RENEWAL
from pd_groundtruth.review_db import PairInsert
from pd_groundtruth.review_db import ReviewDb
from pd_matcher.cli import _load_default_matching_config
from pd_matcher.cli import _load_default_pairing_config
from pd_matcher.config.schemas import MatchingConfig
from pd_matcher.config.schemas import PairingConfig
from pd_matcher.index.lookup import NyplIndexLookup
from pd_matcher.match.combiners.base import CombinedScore
from pd_matcher.match.combiners.calibrator import PlattCalibrator
from pd_matcher.match.evidence import Evidence
from pd_matcher.match.idf import IdfTable
from pd_matcher.match.pairing_compiler import CompiledPairings
from pd_matcher.match.pipeline import _build_context
from pd_matcher.match.scorers.context import ScorerContext
from pd_matcher.models import IndexedNyplRegRecord
from pd_matcher.models import MarcRecord
from pd_matcher.models import NyplRenRecord

_MARC_NS = "http://www.loc.gov/MARC21/slim"
_MARCXML = (
    '<collection xmlns="{ns}">'
    "<record>"
    "<leader>00000nam a2200000 a 4500</leader>"
    '<controlfield tag="001">{control_id}</controlfield>'
    '<controlfield tag="008">750101s1953    xxu           000 0 eng d</controlfield>'
    '<datafield tag="245" ind1="0" ind2="0"><subfield code="a">A Title</subfield></datafield>'
    "</record>"
    "</collection>"
)


def _marc(control_id: str = "ctrl-1", year: int | None = 1953) -> MarcRecord:
    return MarcRecord(
        control_id=control_id,
        title="A Title",
        title_main="A Title",
        main_author="An Author",
        statement_of_responsibility="by An Author",
        publisher="A Publisher",
        publication_year=year,
        language_code="eng",
    )


def _renewal(entry_id: str = "ren-entry-1", odat_year: int = 1953) -> NyplRenRecord:
    return NyplRenRecord(
        id="R200001",
        entry_id=entry_id,
        oreg="A111111",
        odat=date(odat_year, 1, 1),
        rdat=date(1981, 4, 1),
        author="Renewal Author",
        title="Renewal Title",
        claimants="Renewal Claimant",
        new_matter="added chapters",
    )


def _indexed_reg(uuid: str = "reg-uuid", *, was_renewed: bool = False) -> IndexedNyplRegRecord:
    return IndexedNyplRegRecord(
        uuid=uuid, title="Reg Title", was_renewed=was_renewed, reg_year=1953
    )


def _scored(calibrated: float) -> SimpleNamespace:
    """A stand-in for a ``CandidateMatch`` exposing only ``combined.calibrated``."""
    return SimpleNamespace(combined=SimpleNamespace(calibrated=calibrated))


def _evidence(scorer: str, score: float, *, skipped: bool = False) -> Evidence:
    return Evidence(
        scorer=scorer,
        score=score,
        max=1.0,
        skipped=skipped,
        decisive=False,
        features=(),
    )


class _FakeCombiner:
    """A combiner returning a fixed calibrated score for any evidence."""

    def __init__(self, calibrated: float) -> None:
        self._calibrated = calibrated

    def combine(self, evidence: Sequence[Evidence]) -> CombinedScore:
        del evidence
        return CombinedScore(raw=self._calibrated * 100.0, calibrated=self._calibrated)


def _idf() -> IdfTable:
    return IdfTable(document_count=1, default_idf=1.0, source_hash="x", language="eng", idf={})


def _ctx(marc: MarcRecord | None = None) -> ScorerContext:
    return _build_context(marc or _marc(), _idf(), _idf(), _idf(), _config())


def _calibrator() -> PlattCalibrator:
    return PlattCalibrator(
        a=-1.0, b=0.0, trained_at="2026-06-20T00:00:00+00:00", n_positive=1, n_negative=1
    )


def _config() -> MatchingConfig:
    return _load_default_matching_config()


def _pairing_config() -> PairingConfig:
    return _load_default_pairing_config()


# --------------------------------------------------------------------------- #
# score_renewal
# --------------------------------------------------------------------------- #


def test_score_renewal_builds_payload_from_fired_scorers(monkeypatch: MonkeyPatch) -> None:
    """Only non-skipped scorers reach the evidence payload, keyed by group name."""
    monkeypatch.setattr(module, "score_title", lambda *_a: _evidence("title", 0.9))
    monkeypatch.setattr(module, "score_author", lambda *_a: _evidence("author", 0.4))
    monkeypatch.setattr(
        module, "score_publisher", lambda *_a: _evidence("publisher", 0.0, skipped=True)
    )
    monkeypatch.setattr(module, "score_year", lambda *_a: _evidence("year", 1.0))
    result = score_renewal(_marc(), _renewal(), _ctx(), _FakeCombiner(0.77), None)
    assert result.calibrated == 0.77
    assert result.evidence == {"title": 0.9, "author": 0.4, "year": 1.0}


def test_score_renewal_applies_calibrator(monkeypatch: MonkeyPatch) -> None:
    """When a calibrator is supplied the raw score is mapped through it."""
    monkeypatch.setattr(module, "score_title", lambda *_a: _evidence("title", 0.5))
    monkeypatch.setattr(module, "score_author", lambda *_a: _evidence("author", 0.5))
    monkeypatch.setattr(module, "score_publisher", lambda *_a: _evidence("publisher", 0.5))
    monkeypatch.setattr(module, "score_year", lambda *_a: _evidence("year", 0.5))
    monkeypatch.setattr(module, "calibrate", lambda raw, _cal: raw / 200.0)
    result = score_renewal(_marc(), _renewal(), _ctx(), _FakeCombiner(0.6), _calibrator())
    # raw = 0.6 * 100 = 60; calibrate -> 0.3
    assert result.calibrated == 0.3


def test_score_renewal_falls_back_when_no_marc_title_or_author(
    monkeypatch: MonkeyPatch,
) -> None:
    """A MARC with no title/author fields still scores via the fallback call."""
    monkeypatch.setattr(module, "score_title", lambda *_a: _evidence("title", 0.0, skipped=True))
    monkeypatch.setattr(module, "score_author", lambda *_a: _evidence("author", 0.0, skipped=True))
    monkeypatch.setattr(module, "score_publisher", lambda *_a: _evidence("publisher", 0.0))
    monkeypatch.setattr(module, "score_year", lambda *_a: _evidence("year", 0.0))
    bare = MarcRecord(
        control_id="bare",
        title="",
        title_main="",
        publication_year=1953,
        language_code="eng",
    )
    result = score_renewal(bare, _renewal(), _ctx(bare), _FakeCombiner(0.1), None)
    assert result.evidence == {"claimants": 0.0, "year": 0.0}


def test_score_renewal_handles_renewal_without_odat(monkeypatch: MonkeyPatch) -> None:
    """A renewal lacking ``odat`` passes ``None`` as the year to the year scorer."""
    captured: dict[str, object] = {}

    def fake_year(marc_year: object, renewal_year: object, _ctx: object) -> Evidence:
        captured["renewal_year"] = renewal_year
        return _evidence("year", 0.0, skipped=True)

    monkeypatch.setattr(module, "score_title", lambda *_a: _evidence("title", 0.5))
    monkeypatch.setattr(module, "score_author", lambda *_a: _evidence("author", 0.5))
    monkeypatch.setattr(module, "score_publisher", lambda *_a: _evidence("publisher", 0.5))
    monkeypatch.setattr(module, "score_year", fake_year)
    renewal = NyplRenRecord(id="R", entry_id="e", title="T", claimants="C")
    score_renewal(_marc(), renewal, _ctx(), _FakeCombiner(0.5), None)
    assert captured["renewal_year"] is None


# --------------------------------------------------------------------------- #
# best_renewal / _best_evidence
# --------------------------------------------------------------------------- #


def test_best_renewal_returns_none_without_candidates() -> None:
    assert best_renewal(_marc(), (), lambda _m, _r: RenewalScore(0.9, {})) is None


def test_best_renewal_picks_highest_calibrated() -> None:
    low = _renewal(entry_id="low")
    high = _renewal(entry_id="high")
    scores = {"low": 0.3, "high": 0.8}
    result = best_renewal(
        _marc(),
        (low, high),
        lambda _m, renewal: RenewalScore(scores[renewal.entry_id], {}),
    )
    assert result is not None
    renewal, score = result
    assert renewal.entry_id == "high"
    assert score.calibrated == 0.8


def test_best_renewal_keeps_first_when_later_candidate_is_worse() -> None:
    high = _renewal(entry_id="high")
    low = _renewal(entry_id="low")
    scores = {"high": 0.8, "low": 0.3}
    result = best_renewal(
        _marc(),
        (high, low),
        lambda _m, renewal: RenewalScore(scores[renewal.entry_id], {}),
    )
    assert result is not None
    renewal, _score = result
    assert renewal.entry_id == "high"


def test_best_evidence_picks_higher_scoring_candidate() -> None:
    low = _evidence("title", 0.2)
    high = _evidence("title", 0.8)
    assert module._best_evidence((low, high)) is high


# --------------------------------------------------------------------------- #
# _build_renewal_pair_insert
# --------------------------------------------------------------------------- #


def test_build_renewal_pair_insert_maps_renewal_fields() -> None:
    pair = module._build_renewal_pair_insert(
        _marc(),
        _renewal(),
        RenewalScore(0.85, {"title": 0.9, "claimants": 0.5}),
        language="eng",
        band="b80_90",
        audit_note="scenario 4: renewal-only (no registration in odat year 1953)",
    )
    assert pair.audit_note == "scenario 4: renewal-only (no registration in odat year 1953)"
    assert pair.pairing_type == PAIRING_RENEWAL
    assert pair.source == module.SOURCE_RENEWAL
    assert pair.band == "b80_90"
    assert pair.nypl_uuid == "ren-entry-1"
    assert pair.cce_regnum is None
    assert pair.cce_publishers is None
    assert pair.cce_was_renewed is True
    assert pair.cce_title == "Renewal Title"
    assert pair.cce_renewal_id == "R200001"
    assert pair.cce_renewal_oreg == "A111111"
    assert pair.cce_renewal_rdat == "1981-04-01"
    assert pair.cce_reg_year == 1953
    assert pair.evidence_json == '{"title":0.9,"claimants":0.5}'


def test_build_renewal_pair_insert_handles_renewal_without_odat_or_rdat() -> None:
    renewal = NyplRenRecord(id="R", entry_id="e", title="T", claimants="C")
    pair = module._build_renewal_pair_insert(
        _marc(),
        renewal,
        RenewalScore(0.7, {}),
        language="eng",
        band="b60_70",
        audit_note="scenario 4: renewal-only (no registration in odat year None)",
    )
    assert pair.cce_reg_year is None
    assert pair.cce_renewal_rdat is None


# --------------------------------------------------------------------------- #
# _scenario_4_note
# --------------------------------------------------------------------------- #


def test_scenario_4_note_names_odat_year() -> None:
    assert (
        module._scenario_4_note(1949)
        == "scenario 4: renewal-only (no registration in odat year 1949)"
    )


def test_scenario_4_note_handles_missing_year() -> None:
    assert (
        module._scenario_4_note(None)
        == "scenario 4: renewal-only (no registration in odat year None)"
    )


# --------------------------------------------------------------------------- #
# _make_reg_present_fn
# --------------------------------------------------------------------------- #


class _FakeLookup:
    """A stand-in for ``NyplIndexLookup`` serving canned renewal and registration data.

    ``by_marc`` maps a MARC control id to its renewal candidates. ``reg_by_year``
    maps a ``(control_id, year)`` pair to the registration candidates returned by
    :meth:`candidates_in_year`, so a test can prove the registration check keys
    off an explicit year. Every ``candidates_in_year`` call is recorded.
    """

    def __init__(
        self,
        by_marc: dict[str, tuple[NyplRenRecord, ...]],
        reg_by_year: dict[tuple[str, int], tuple[IndexedNyplRegRecord, ...]] | None = None,
    ) -> None:
        self._by_marc = by_marc
        self._reg_by_year = reg_by_year or {}
        self.candidates_in_year_calls: list[tuple[str, int]] = []

    def __enter__(self) -> _FakeLookup:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def candidates_for_renewal(self, marc: MarcRecord, _window: int) -> tuple[NyplRenRecord, ...]:
        return self._by_marc.get(marc.control_id, ())

    def candidates_in_year(
        self, marc: MarcRecord, year: int, _window: int = 0
    ) -> tuple[IndexedNyplRegRecord, ...]:
        self.candidates_in_year_calls.append((marc.control_id, year))
        return self._reg_by_year.get((marc.control_id, year), ())


def _reg_present_fn(
    monkeypatch: MonkeyPatch,
    lookup: _FakeLookup,
    *,
    reg_scores: dict[str, float],
    reg_min_calibrated: float,
) -> module.RegPresentFn:
    monkeypatch.setattr(
        module,
        "_score_candidate",
        lambda _m, candidate, *_a, **_k: _scored(reg_scores.get(candidate.uuid, 0.99)),
    )
    return module._make_reg_present_fn(
        cast(NyplIndexLookup, lookup),
        _config(),
        _idf(),
        _idf(),
        _idf(),
        None,
        _FakeCombiner(0.0),
        cast(CompiledPairings, object()),
        reg_min_calibrated,
    )


def test_reg_present_fn_returns_false_for_none_year(monkeypatch: MonkeyPatch) -> None:
    """A renewal without an ``odat`` year cannot be checked and is not present."""
    lookup = _FakeLookup({})
    fn = _reg_present_fn(monkeypatch, lookup, reg_scores={}, reg_min_calibrated=0.9)
    assert fn(_marc(), None) is False
    assert lookup.candidates_in_year_calls == []


def test_reg_present_fn_true_when_candidate_clears_floor(monkeypatch: MonkeyPatch) -> None:
    lookup = _FakeLookup({}, {("ctrl-1", 1953): (_indexed_reg("u1"),)})
    fn = _reg_present_fn(monkeypatch, lookup, reg_scores={"u1": 0.95}, reg_min_calibrated=0.9)
    assert fn(_marc(), 1953) is True
    assert lookup.candidates_in_year_calls == [("ctrl-1", 1953)]


def test_reg_present_fn_false_when_candidate_below_floor(monkeypatch: MonkeyPatch) -> None:
    lookup = _FakeLookup({}, {("ctrl-1", 1953): (_indexed_reg("u1"),)})
    fn = _reg_present_fn(monkeypatch, lookup, reg_scores={"u1": 0.5}, reg_min_calibrated=0.9)
    assert fn(_marc(), 1953) is False


def test_reg_present_fn_caches_context_per_marc(monkeypatch: MonkeyPatch) -> None:
    """The per-MARC context is built once and reused across both year lookups."""
    calls: list[str] = []

    def fake_build_context(marc: MarcRecord, *_a: object) -> object:
        calls.append(marc.control_id)
        return object()

    monkeypatch.setattr(module, "_build_context", fake_build_context)
    lookup = _FakeLookup(
        {}, {("ctrl-1", 1953): (_indexed_reg("u1"),), ("ctrl-1", 1954): (_indexed_reg("u2"),)}
    )
    fn = _reg_present_fn(
        monkeypatch, lookup, reg_scores={"u1": 0.0, "u2": 0.0}, reg_min_calibrated=0.9
    )
    marc = _marc()
    fn(marc, 1953)
    fn(marc, 1954)
    assert calls == ["ctrl-1"]


# --------------------------------------------------------------------------- #
# _make_score_fn
# --------------------------------------------------------------------------- #


def test_make_score_fn_caches_context_per_marc(monkeypatch: MonkeyPatch) -> None:
    """The per-MARC context is built once and reused across that MARC's candidates."""
    calls: list[str] = []

    def fake_build_context(marc: MarcRecord, *_a: object) -> object:
        calls.append(marc.control_id)
        return object()

    monkeypatch.setattr(module, "_build_context", fake_build_context)
    monkeypatch.setattr(module, "score_renewal", lambda *_a: RenewalScore(0.5, {}))
    score_fn = module._make_score_fn(_idf(), _idf(), _idf(), _config(), _FakeCombiner(0.5), None)
    marc = _marc()
    score_fn(marc, _renewal("a"))
    score_fn(marc, _renewal("b"))
    score_fn(_marc("other"), _renewal("c"))
    assert calls == ["ctrl-1", "other"]


# --------------------------------------------------------------------------- #
# _load_calibrator
# --------------------------------------------------------------------------- #


def test_load_calibrator_returns_none_when_absent(tmp_path: Path) -> None:
    assert module._load_calibrator(tmp_path) is None


def test_load_calibrator_loads_when_present(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    (tmp_path / "calibrator.msgpack").write_bytes(b"unused")
    calibrator = _calibrator()
    monkeypatch.setattr(module, "load_calibrator", lambda _path: calibrator)
    assert module._load_calibrator(tmp_path) is calibrator


# --------------------------------------------------------------------------- #
# build_renewal_queue (renewal-first integration)
# --------------------------------------------------------------------------- #


def _write_pool(pool: Path, control_ids: tuple[str, ...]) -> None:
    lang_dir = pool / "eng"
    lang_dir.mkdir(parents=True)
    for index, control_id in enumerate(control_ids):
        (lang_dir / f"shard_{index}.xml").write_text(
            _MARCXML.format(ns=_MARC_NS, control_id=control_id), encoding="utf-8"
        )


def _patch_wiring(
    monkeypatch: MonkeyPatch,
    lookup: _FakeLookup,
    renewal_calibrated: float,
    *,
    reg_scores: dict[str, float] | None = None,
) -> None:
    monkeypatch.setattr(module, "load_or_build_idf", lambda *_a, **_k: object())
    monkeypatch.setattr(module, "load_or_build_author_idf", lambda *_a, **_k: object())
    monkeypatch.setattr(module, "load_or_build_publisher_idf", lambda *_a, **_k: object())
    monkeypatch.setattr(module, "_load_calibrator", lambda _parent: None)
    monkeypatch.setattr(
        module, "build_combiner", lambda _c, **_k: _FakeCombiner(renewal_calibrated)
    )
    monkeypatch.setattr(module, "compile_pairings", lambda _c: object())
    monkeypatch.setattr(module, "NyplIndexLookup", lambda _path: lookup)
    monkeypatch.setattr(
        module,
        "_make_score_fn",
        lambda *_a, **_k: (
            lambda _m, _r: RenewalScore(renewal_calibrated, {"title": renewal_calibrated})
        ),
    )
    scores = reg_scores or {}
    monkeypatch.setattr(
        module,
        "_score_candidate",
        lambda _m, candidate, *_a, **_k: _scored(scores.get(candidate.uuid, 0.99)),
    )


def _run(tmp_path: Path) -> module.RenewalBuildSummary:
    return build_renewal_queue(
        pool=tmp_path / "pool",
        index_path=tmp_path / "index.lmdb",
        out_path=tmp_path / "review.db",
        matching_config=_config(),
        pairing_config=_pairing_config(),
        min_score=60.0,
        reg_min_score=90.0,
        reg_scorer="learned",
    )


def test_build_renewal_queue_emits_scenario_4_when_no_registration(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    """A renewal-haver with no registration in the odat year yields a scenario-4 pair."""
    _write_pool(tmp_path / "pool", ("ctrl-a",))
    lookup = _FakeLookup({"ctrl-a": (_renewal("ea", odat_year=1953),)})
    _patch_wiring(monkeypatch, lookup, 0.9)
    summary = _run(tmp_path)
    assert summary.records_scanned == 1
    assert summary.renewal_havers == 1
    assert summary.reg_excluded == 0
    assert summary.scenario4_written == 1
    assert lookup.candidates_in_year_calls == [("ctrl-a", 1953)]
    with ReviewDb.connect(tmp_path / "review.db") as db:
        pair = db.get_pair(1)
    assert pair is not None
    assert pair.pairing_type == PAIRING_RENEWAL
    assert pair.audit_note == "scenario 4: renewal-only (no registration in odat year 1953)"


def test_build_renewal_queue_excludes_when_registration_found(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    """A renewal-haver WITH a registration at or above the floor is excluded."""
    _write_pool(tmp_path / "pool", ("ctrl-a",))
    lookup = _FakeLookup(
        {"ctrl-a": (_renewal("ea", odat_year=1953),)},
        {("ctrl-a", 1953): (_indexed_reg("u1"),)},
    )
    _patch_wiring(monkeypatch, lookup, 0.9)
    summary = _run(tmp_path)
    assert summary.renewal_havers == 1
    assert summary.reg_excluded == 1
    assert summary.scenario4_written == 0
    with ReviewDb.connect(tmp_path / "review.db") as db:
        assert db.get_pair(1) is None


def test_build_renewal_queue_emits_when_registration_below_floor(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    """A registration present but below the reg floor does not exclude the book."""
    _write_pool(tmp_path / "pool", ("ctrl-a",))
    lookup = _FakeLookup(
        {"ctrl-a": (_renewal("ea", odat_year=1953),)},
        {("ctrl-a", 1953): (_indexed_reg("u1"),)},
    )
    _patch_wiring(monkeypatch, lookup, 0.9, reg_scores={"u1": 0.5})
    summary = _run(tmp_path)
    assert summary.reg_excluded == 0
    assert summary.scenario4_written == 1


def test_build_renewal_queue_reg_check_uses_odat_year_not_pub_year(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    """The registration check keys off the renewal's odat year, not the MARC pub year.

    The MARC's publication year is 1953 (from the 008 fixture) but the best
    renewal's ``odat`` year is 1949. A registration planted only at 1949 must
    exclude the book, proving the check looked at the odat year.
    """
    _write_pool(tmp_path / "pool", ("ctrl-a",))
    lookup = _FakeLookup(
        {"ctrl-a": (_renewal("ea", odat_year=1949),)},
        {("ctrl-a", 1949): (_indexed_reg("u1"),)},
    )
    _patch_wiring(monkeypatch, lookup, 0.9)
    summary = _run(tmp_path)
    assert summary.reg_excluded == 1
    assert summary.scenario4_written == 0
    assert lookup.candidates_in_year_calls == [("ctrl-a", 1949)]


def test_build_renewal_queue_ignores_registration_in_pub_year(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    """A registration only in the MARC pub year (not the odat year) is ignored."""
    _write_pool(tmp_path / "pool", ("ctrl-a",))
    lookup = _FakeLookup(
        {"ctrl-a": (_renewal("ea", odat_year=1949),)},
        {("ctrl-a", 1953): (_indexed_reg("u1"),)},
    )
    _patch_wiring(monkeypatch, lookup, 0.9)
    summary = _run(tmp_path)
    assert summary.reg_excluded == 0
    assert summary.scenario4_written == 1
    assert lookup.candidates_in_year_calls == [("ctrl-a", 1949)]
    with ReviewDb.connect(tmp_path / "review.db") as db:
        pair = db.get_pair(1)
    assert pair is not None
    assert pair.audit_note == "scenario 4: renewal-only (no registration in odat year 1949)"


def test_build_renewal_queue_skips_non_renewal_haver_without_running_reg_check(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    """A best renewal below the floor is skipped and the reg check never runs (efficiency)."""
    _write_pool(tmp_path / "pool", ("ctrl-a",))
    lookup = _FakeLookup(
        {"ctrl-a": (_renewal("ea", odat_year=1953),)},
        {("ctrl-a", 1953): (_indexed_reg("u1"),)},
    )
    reg_calls: list[str] = []
    _patch_wiring(monkeypatch, lookup, 0.3)

    def _recording_score(
        _m: object, candidate: IndexedNyplRegRecord, *_a: object, **_k: object
    ) -> SimpleNamespace:
        reg_calls.append(candidate.uuid)
        return _scored(0.99)

    monkeypatch.setattr(module, "_score_candidate", _recording_score)
    summary = _run(tmp_path)
    assert summary.records_scanned == 1
    assert summary.renewal_havers == 0
    assert summary.scenario4_written == 0
    assert reg_calls == []
    assert lookup.candidates_in_year_calls == []


def test_build_renewal_queue_skips_marc_without_renewal_candidates(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    """A MARC with no renewal candidates at all is skipped before the reg check."""
    _write_pool(tmp_path / "pool", ("ctrl-a",))
    lookup = _FakeLookup({})
    _patch_wiring(monkeypatch, lookup, 0.9)
    summary = _run(tmp_path)
    assert summary.records_scanned == 1
    assert summary.renewal_havers == 0
    assert summary.scenario4_written == 0
    assert lookup.candidates_in_year_calls == []


def test_build_renewal_queue_writes_multiple_pairs_and_commits(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    """Two scenario-4 books are both written; the fill-interval commit fires."""
    _write_pool(tmp_path / "pool", ("ctrl-a", "ctrl-b"))
    lookup = _FakeLookup(
        {"ctrl-a": (_renewal("ea", odat_year=1953),), "ctrl-b": (_renewal("eb", odat_year=1953),)}
    )
    _patch_wiring(monkeypatch, lookup, 0.9)
    monkeypatch.setattr(module, "_FILL_LOG_INTERVAL", 1)
    summary = _run(tmp_path)
    assert summary.records_scanned == 2
    assert summary.scenario4_written == 2
    with ReviewDb.connect(tmp_path / "review.db") as db:
        rows = [db.get_pair(1), db.get_pair(2)]
    assert all(row is not None and row.pairing_type == PAIRING_RENEWAL for row in rows)


def test_build_renewal_queue_skips_marcs_already_in_db(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    _write_pool(tmp_path / "pool", ("ctrl-a", "ctrl-b"))
    out = tmp_path / "review.db"
    with ReviewDb.connect(out) as db:
        db.insert_pair(_existing_registration_pair("ctrl-a"))
    lookup = _FakeLookup(
        {"ctrl-a": (_renewal("ea", odat_year=1953),), "ctrl-b": (_renewal("eb", odat_year=1953),)}
    )
    _patch_wiring(monkeypatch, lookup, 0.9)
    summary = _run(tmp_path)
    assert summary.records_scanned == 1
    assert summary.scenario4_written == 1
    assert lookup.candidates_in_year_calls == [("ctrl-b", 1953)]


def test_build_renewal_queue_persists_rows_on_interrupt(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    """A Ctrl-C mid-build must keep rows inserted before the interrupt.

    insert_pair does not commit and ReviewDb.__exit__ only commits on a clean
    exit, so without the finally-commit a KeyboardInterrupt would roll back the
    whole queue. This drives the pool iterator to raise after two records and
    asserts both survive.
    """
    _write_pool(tmp_path / "pool", ("ctrl-a", "ctrl-b"))
    lookup = _FakeLookup(
        {"ctrl-a": (_renewal("ea", odat_year=1953),), "ctrl-b": (_renewal("eb", odat_year=1953),)}
    )
    _patch_wiring(monkeypatch, lookup, 0.9)

    def _interrupting_pool(_pool: Path) -> Iterator[MarcRecord]:
        yield _marc("ctrl-a")
        yield _marc("ctrl-b")
        raise KeyboardInterrupt

    monkeypatch.setattr(module, "_iter_pool_records", _interrupting_pool)
    with raises(KeyboardInterrupt):
        _run(tmp_path)
    with ReviewDb.connect(tmp_path / "review.db") as db:
        assert db.get_pair(1) is not None
        assert db.get_pair(2) is not None


def test_build_renewal_queue_logs_scanned_progress(
    tmp_path: Path, monkeypatch: MonkeyPatch, caplog: LogCaptureFixture
) -> None:
    """The scanned-progress log fires on the scanned interval."""
    _write_pool(tmp_path / "pool", ("ctrl-a",))
    lookup = _FakeLookup({"ctrl-a": (_renewal("ea", odat_year=1953),)})
    _patch_wiring(monkeypatch, lookup, 0.9)
    monkeypatch.setattr(module, "_SCANNED_LOG_INTERVAL", 1)
    with caplog.at_level("INFO", logger="pd_groundtruth.build_renewal_queue"):
        _run(tmp_path)
    assert any("renewal queue: scanned=1" in message for message in caplog.messages)


def _existing_registration_pair(control_id: str) -> PairInsert:
    return PairInsert(
        language="eng",
        decade=1950,
        score=0.95,
        band="ge90",
        source="banded",
        marc_control_id=control_id,
        marc_json='{"control_id": "x"}',
        marc_title="A Title",
        marc_author="An Author",
        marc_publisher="A Publisher",
        marc_year=1953,
        nypl_uuid="reg-uuid",
        cce_title="CCE Title",
        cce_author="CCE Author",
        cce_publishers="Pub",
        cce_claimants="Claimant",
        cce_reg_year=1953,
        cce_was_renewed=True,
        cce_regnum="R1",
        evidence_json="{}",
    )
