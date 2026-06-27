"""Unit tests for the renewal-pair review-queue builder."""

from __future__ import annotations

from collections.abc import Iterator
from collections.abc import Sequence
from datetime import date
from pathlib import Path

from pytest import MonkeyPatch
from pytest import raises

from pd_groundtruth import build_renewal_queue as module
from pd_groundtruth.build_renewal_queue import RenewalScore
from pd_groundtruth.build_renewal_queue import best_renewal
from pd_groundtruth.build_renewal_queue import build_renewal_queue
from pd_groundtruth.build_renewal_queue import renewal_pair_for
from pd_groundtruth.build_renewal_queue import score_renewal
from pd_groundtruth.review_db import PAIRING_RENEWAL
from pd_groundtruth.review_db import PairInsert
from pd_groundtruth.review_db import ReviewDb
from pd_matcher.cli import _load_default_matching_config
from pd_matcher.config.schemas import MatchingConfig
from pd_matcher.match.combiners.base import CombinedScore
from pd_matcher.match.combiners.calibrator import PlattCalibrator
from pd_matcher.match.evidence import Evidence
from pd_matcher.match.idf import IdfTable
from pd_matcher.match.pipeline import _build_context
from pd_matcher.match.scorers.context import ScorerContext
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


def test_renewal_pair_for_returns_none_when_below_floor() -> None:
    pair = renewal_pair_for(
        _marc(),
        (_renewal(),),
        score_fn=lambda _m, _r: RenewalScore(0.4, {"title": 0.4}),
        min_calibrated=0.6,
    )
    assert pair is None


def test_renewal_pair_for_returns_none_when_no_candidate() -> None:
    pair = renewal_pair_for(
        _marc(),
        (),
        score_fn=lambda _m, _r: RenewalScore(0.9, {}),
        min_calibrated=0.6,
    )
    assert pair is None


def test_renewal_pair_for_builds_renewal_pair_insert() -> None:
    pair = renewal_pair_for(
        _marc(),
        (_renewal(),),
        score_fn=lambda _m, _r: RenewalScore(0.85, {"title": 0.9, "claimants": 0.5}),
        min_calibrated=0.6,
    )
    assert pair is not None
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


def test_renewal_pair_for_handles_renewal_without_odat_or_rdat() -> None:
    renewal = NyplRenRecord(id="R", entry_id="e", title="T", claimants="C")
    pair = renewal_pair_for(
        _marc(),
        (renewal,),
        score_fn=lambda _m, _r: RenewalScore(0.7, {}),
        min_calibrated=0.6,
    )
    assert pair is not None
    assert pair.cce_reg_year is None
    assert pair.cce_renewal_rdat is None


class _FakeLookup:
    """A stand-in for ``NyplIndexLookup`` returning canned renewal candidates."""

    def __init__(self, by_marc: dict[str, tuple[NyplRenRecord, ...]]) -> None:
        self._by_marc = by_marc

    def __enter__(self) -> _FakeLookup:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def candidates_for_renewal(self, marc: MarcRecord, _window: int) -> tuple[NyplRenRecord, ...]:
        return self._by_marc.get(marc.control_id, ())


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
    calibrated: float,
) -> None:
    monkeypatch.setattr(module, "load_or_build_idf", lambda *_a, **_k: object())
    monkeypatch.setattr(module, "load_or_build_author_idf", lambda *_a, **_k: object())
    monkeypatch.setattr(module, "load_or_build_publisher_idf", lambda *_a, **_k: object())
    monkeypatch.setattr(module, "_load_calibrator", lambda _parent: None)
    monkeypatch.setattr(module, "build_combiner", lambda _c, **_k: _FakeCombiner(calibrated))
    monkeypatch.setattr(module, "NyplIndexLookup", lambda _path: lookup)
    monkeypatch.setattr(
        module,
        "_make_score_fn",
        lambda *_a, **_k: lambda _m, _r: RenewalScore(calibrated, {"title": calibrated}),
    )


def _config() -> MatchingConfig:
    return _load_default_matching_config()


def test_build_renewal_queue_writes_pairs_above_floor(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    pool = tmp_path / "pool"
    _write_pool(pool, ("ctrl-a", "ctrl-b"))
    lookup = _FakeLookup({"ctrl-a": (_renewal("ea"),), "ctrl-b": (_renewal("eb"),)})
    _patch_wiring(monkeypatch, lookup, 0.9)
    # Force the periodic fill log to fire on the second write.
    monkeypatch.setattr(module, "_FILL_LOG_INTERVAL", 2)
    out = tmp_path / "review.db"
    summary = build_renewal_queue(
        pool=pool,
        index_path=tmp_path / "index.lmdb",
        out_path=out,
        matching_config=_config(),
        min_score=60.0,
    )
    assert summary.records_scanned == 2
    assert summary.pairs_written == 2
    with ReviewDb.connect(out) as db:
        rows = [db.get_pair(1), db.get_pair(2)]
    assert all(row is not None and row.pairing_type == PAIRING_RENEWAL for row in rows)


def test_build_renewal_queue_persists_rows_on_interrupt(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    """A Ctrl-C mid-build must keep rows inserted before the interrupt.

    insert_pair does not commit and ReviewDb.__exit__ only commits on a clean
    exit, so without the finally-commit a KeyboardInterrupt would roll back the
    whole queue. This drives the pool iterator to raise after two records and
    asserts both survive.
    """
    pool = tmp_path / "pool"
    _write_pool(pool, ("ctrl-a", "ctrl-b"))
    lookup = _FakeLookup({"ctrl-a": (_renewal("ea"),), "ctrl-b": (_renewal("eb"),)})
    _patch_wiring(monkeypatch, lookup, 0.9)

    def _interrupting_pool(_pool: Path) -> Iterator[MarcRecord]:
        yield _marc("ctrl-a")
        yield _marc("ctrl-b")
        raise KeyboardInterrupt

    monkeypatch.setattr(module, "_iter_pool_records", _interrupting_pool)
    out = tmp_path / "review.db"
    with raises(KeyboardInterrupt):
        build_renewal_queue(
            pool=pool,
            index_path=tmp_path / "index.lmdb",
            out_path=out,
            matching_config=_config(),
            min_score=60.0,
        )
    with ReviewDb.connect(out) as db:
        assert db.get_pair(1) is not None
        assert db.get_pair(2) is not None


def test_build_renewal_queue_skips_below_floor(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    pool = tmp_path / "pool"
    _write_pool(pool, ("ctrl-a",))
    lookup = _FakeLookup({"ctrl-a": (_renewal("ea"),)})
    _patch_wiring(monkeypatch, lookup, 0.3)
    out = tmp_path / "review.db"
    summary = build_renewal_queue(
        pool=pool,
        index_path=tmp_path / "index.lmdb",
        out_path=out,
        matching_config=_config(),
        min_score=60.0,
    )
    assert summary.records_scanned == 1
    assert summary.pairs_written == 0


def test_build_renewal_queue_skips_marcs_already_in_db(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    pool = tmp_path / "pool"
    _write_pool(pool, ("ctrl-a", "ctrl-b"))
    out = tmp_path / "review.db"
    # ctrl-a is already queued (as a registration pair) and must not be redone.
    with ReviewDb.connect(out) as db:
        db.insert_pair(_existing_registration_pair("ctrl-a"))
    lookup = _FakeLookup({"ctrl-a": (_renewal("ea"),), "ctrl-b": (_renewal("eb"),)})
    _patch_wiring(monkeypatch, lookup, 0.9)
    summary = build_renewal_queue(
        pool=pool,
        index_path=tmp_path / "index.lmdb",
        out_path=out,
        matching_config=_config(),
        min_score=60.0,
    )
    assert summary.records_scanned == 1
    assert summary.pairs_written == 1


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


def test_best_evidence_picks_higher_scoring_candidate() -> None:
    low = _evidence("title", 0.2)
    high = _evidence("title", 0.8)
    assert module._best_evidence((low, high)) is high


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


def test_load_calibrator_returns_none_when_absent(tmp_path: Path) -> None:
    assert module._load_calibrator(tmp_path) is None


def test_load_calibrator_loads_when_present(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    (tmp_path / "calibrator.msgpack").write_bytes(b"unused")
    calibrator = _calibrator()
    monkeypatch.setattr(module, "load_calibrator", lambda _path: calibrator)
    assert module._load_calibrator(tmp_path) is calibrator
