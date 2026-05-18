"""Tests for :mod:`pd_matcher.match.pipeline`."""

from pathlib import Path

from pd_matcher.config.schemas import MatchingConfig
from pd_matcher.index.builder import build_index
from pd_matcher.index.lookup import NyplIndexLookup
from pd_matcher.match.combiners.calibrator import PlattCalibrator
from pd_matcher.match.combiners.weighted_mean import WeightedMeanCombiner
from pd_matcher.match.idf import IdfTable
from pd_matcher.match.idf import build_idf_table
from pd_matcher.match.pipeline import match_record
from pd_matcher.models import MarcRecord

_FIXTURES = Path(__file__).resolve().parents[2] / "fixtures"


def _build_tiny_index(root: Path) -> Path:
    reg_dir = root / "reg"
    ren_dir = root / "ren"
    reg_dir.mkdir()
    ren_dir.mkdir()
    (reg_dir / "tiny_reg.xml").write_bytes((_FIXTURES / "tiny_reg.xml").read_bytes())
    (ren_dir / "tiny_ren.tsv").write_bytes((_FIXTURES / "tiny_ren.tsv").read_bytes())
    out_path = root / "idx.lmdb"
    build_index(reg_dir=reg_dir, ren_dir=ren_dir, out_path=out_path)
    return out_path


def _idf(lookup: NyplIndexLookup) -> IdfTable:
    return build_idf_table(lookup)


def _config(*, min_score: float = 30.0) -> MatchingConfig:
    return MatchingConfig(
        title_weight=0.40,
        author_weight=0.20,
        publisher_weight=0.10,
        year_weight=0.10,
        edition_weight=0.05,
        lccn_weight=0.10,
        isbn_weight=0.05,
        year_window=2,
        min_combined_score=min_score,
        scorer="weighted_mean",
    )


def test_match_record_returns_empty_when_marc_has_no_year(tmp_path: Path) -> None:
    """A MARC record with no year is returned with no candidates considered."""
    out_path = _build_tiny_index(tmp_path)
    with NyplIndexLookup(out_path) as lookup:
        idf = _idf(lookup)
        marc = MarcRecord(control_id="m", title="A study of widgets")
        result = match_record(
            marc,
            lookup=lookup,
            config=_config(),
            idf=idf,
            calibrator=None,
            combiner=WeightedMeanCombiner(config=_config()),
        )
    assert result.best is None
    assert result.alternates == ()
    assert result.candidates_considered == 0


def test_match_record_returns_empty_when_no_candidates_in_year_window(tmp_path: Path) -> None:
    """No candidates in the year bucket → empty result."""
    out_path = _build_tiny_index(tmp_path)
    with NyplIndexLookup(out_path) as lookup:
        idf = _idf(lookup)
        marc = MarcRecord(
            control_id="m",
            title="A study of widgets",
            publication_year=1800,
        )
        result = match_record(
            marc,
            lookup=lookup,
            config=_config(),
            idf=idf,
            calibrator=None,
            combiner=WeightedMeanCombiner(config=_config()),
        )
    assert result.best is None
    assert result.candidates_considered == 0


def test_match_record_picks_uuid_0001_for_widget_study(tmp_path: Path) -> None:
    """The widget-study MARC record should match UUID-0001 in the fixture."""
    out_path = _build_tiny_index(tmp_path)
    with NyplIndexLookup(out_path) as lookup:
        idf = _idf(lookup)
        config = _config()
        marc = MarcRecord(
            control_id="m",
            title="A study of widgets",
            main_author="Smith, John",
            publisher="Acme Press",
            edition="1st ed.",
            publication_year=1940,
        )
        result = match_record(
            marc,
            lookup=lookup,
            config=config,
            idf=idf,
            calibrator=None,
            combiner=WeightedMeanCombiner(config=config),
        )
    assert result.best is not None
    assert result.best.nypl_uuid == "UUID-0001"
    assert result.best.combined.raw > 70.0
    assert result.candidates_considered >= 1


def test_match_record_returns_no_best_when_below_floor(tmp_path: Path) -> None:
    """A high min_combined_score floor filters all candidates out."""
    out_path = _build_tiny_index(tmp_path)
    with NyplIndexLookup(out_path) as lookup:
        idf = _idf(lookup)
        config = _config(min_score=99.0)
        marc = MarcRecord(
            control_id="m",
            title="Completely unrelated title",
            publication_year=1940,
        )
        result = match_record(
            marc,
            lookup=lookup,
            config=config,
            idf=idf,
            calibrator=None,
            combiner=WeightedMeanCombiner(config=config),
        )
    assert result.best is None
    assert result.candidates_considered >= 1


def test_match_record_applies_calibrator(tmp_path: Path) -> None:
    """When a calibrator is supplied, calibrated overrides raw/100."""
    out_path = _build_tiny_index(tmp_path)
    with NyplIndexLookup(out_path) as lookup:
        idf = _idf(lookup)
        config = _config(min_score=0.0)
        calibrator = PlattCalibrator(
            a=-0.1,
            b=5.0,
            trained_at="2026-01-01T00:00:00+00:00",
            n_positive=1,
            n_negative=1,
        )
        marc = MarcRecord(
            control_id="m",
            title="A study of widgets",
            publication_year=1940,
        )
        result = match_record(
            marc,
            lookup=lookup,
            config=config,
            idf=idf,
            calibrator=calibrator,
            combiner=WeightedMeanCombiner(config=config),
        )
    assert result.best is not None
    # Calibrated value should not equal raw/100 thanks to the supplied params.
    assert result.best.combined.calibrated != result.best.combined.raw / 100.0


def test_match_record_returns_alternates_in_descending_calibrated_order(tmp_path: Path) -> None:
    """When multiple candidates pass the floor, alternates are sorted high→low."""
    out_path = _build_tiny_index(tmp_path)
    with NyplIndexLookup(out_path) as lookup:
        idf = _idf(lookup)
        config = _config(min_score=0.0)
        marc = MarcRecord(
            control_id="m",
            title="A study of widgets",
            publication_year=1940,
        )
        result = match_record(
            marc,
            lookup=lookup,
            config=config,
            idf=idf,
            calibrator=None,
            combiner=WeightedMeanCombiner(config=config),
            top_k=3,
        )
    if result.alternates:
        assert result.best is not None
        scores = [
            result.best.combined.calibrated,
            *(alt.combined.calibrated for alt in result.alternates),
        ]
        assert scores == sorted(scores, reverse=True)


def test_match_record_prefers_series_title_pairing_when_it_scores_higher(
    tmp_path: Path,
) -> None:
    """A series title that perfectly matches NYPL wins over a primary mismatch."""
    out_path = _build_tiny_index(tmp_path)
    with NyplIndexLookup(out_path) as lookup:
        idf = _idf(lookup)
        config = _config(min_score=0.0)
        # marc.title is unrelated; the series_titles contain the real match.
        marc = MarcRecord(
            control_id="m",
            title="Completely unrelated cover title",
            series_titles=("A study of widgets",),
            publication_year=1940,
        )
        result = match_record(
            marc,
            lookup=lookup,
            config=config,
            idf=idf,
            calibrator=None,
            combiner=WeightedMeanCombiner(config=config),
        )
    assert result.best is not None
    title_evidence = next(ev for ev in result.best.evidence if ev.scorer == "title.token_set")
    losing_titles = [ev for ev in result.best.losing_evidence if ev.scorer == "title.token_set"]
    assert title_evidence.score > losing_titles[0].score


def test_match_record_includes_losing_evidence_from_alternate_pairings(tmp_path: Path) -> None:
    """A MARC record with series titles records the losing title pairing."""
    out_path = _build_tiny_index(tmp_path)
    with NyplIndexLookup(out_path) as lookup:
        idf = _idf(lookup)
        config = _config(min_score=0.0)
        marc = MarcRecord(
            control_id="m",
            title="A study of widgets",
            series_titles=("Some other series",),
            publication_year=1940,
        )
        result = match_record(
            marc,
            lookup=lookup,
            config=config,
            idf=idf,
            calibrator=None,
            combiner=WeightedMeanCombiner(config=config),
        )
    assert result.best is not None
    assert len(result.best.losing_evidence) >= 1
