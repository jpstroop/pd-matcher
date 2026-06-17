"""Tests for :mod:`pd_matcher.match.pipeline`."""

from pathlib import Path

from pd_matcher.config.schemas import FieldSpec
from pd_matcher.config.schemas import MatchingConfig
from pd_matcher.config.schemas import PairingConfig
from pd_matcher.config.schemas import PairingSpec
from pd_matcher.index.builder import build_index
from pd_matcher.index.lookup import NyplIndexLookup
from pd_matcher.match.combiners.calibrator import PlattCalibrator
from pd_matcher.match.combiners.weighted_mean import WeightedMeanCombiner
from pd_matcher.match.evidence import Evidence
from pd_matcher.match.idf import IdfTable
from pd_matcher.match.idf import build_idf_table
from pd_matcher.match.pairing_compiler import CompiledPairings
from pd_matcher.match.pairing_compiler import compile_pairings
from pd_matcher.match.pipeline import _apply_title_isolation_multiplier
from pd_matcher.match.pipeline import _with_multiplier
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


def _config(*, min_score: float = 30.0, year_window: int = 2) -> MatchingConfig:
    return MatchingConfig(
        title_weight=0.40,
        author_weight=0.20,
        publisher_weight=0.10,
        year_weight=0.10,
        edition_weight=0.05,
        lccn_weight=0.10,
        isbn_weight=0.05,
        extent_weight=0.0,
        volume_weight=0.0,
        year_window=year_window,
        min_combined_score=min_score,
        scorer="weighted_mean",
    )


def test_match_record_returns_empty_when_marc_has_no_year(
    tmp_path: Path, compiled_pairings: CompiledPairings
) -> None:
    """A MARC record with no year is returned with no candidates considered."""
    out_path = _build_tiny_index(tmp_path)
    with NyplIndexLookup(out_path) as lookup:
        idf = _idf(lookup)
        marc = MarcRecord(
            control_id="m", title="A study of widgets", title_main="A study of widgets"
        )
        result = match_record(
            marc,
            lookup=lookup,
            config=_config(),
            idf=idf,
            author_idf=idf,
            publisher_idf=idf,
            calibrator=None,
            combiner=WeightedMeanCombiner(config=_config()),
            pairings=compiled_pairings,
        )
    assert result.best is None
    assert result.alternates == ()
    assert result.candidates_considered == 0


def test_match_record_returns_empty_when_no_candidates_in_year_window(
    tmp_path: Path, compiled_pairings: CompiledPairings
) -> None:
    """No candidates in the year bucket → empty result."""
    out_path = _build_tiny_index(tmp_path)
    with NyplIndexLookup(out_path) as lookup:
        idf = _idf(lookup)
        marc = MarcRecord(
            control_id="m",
            title="A study of widgets",
            title_main="A study of widgets",
            publication_year=1800,
        )
        result = match_record(
            marc,
            lookup=lookup,
            config=_config(),
            idf=idf,
            author_idf=idf,
            publisher_idf=idf,
            calibrator=None,
            combiner=WeightedMeanCombiner(config=_config()),
            pairings=compiled_pairings,
        )
    assert result.best is None
    assert result.candidates_considered == 0


def test_match_record_picks_uuid_0001_for_widget_study(
    tmp_path: Path, compiled_pairings: CompiledPairings
) -> None:
    """The widget-study MARC record should match UUID-0001 in the fixture."""
    out_path = _build_tiny_index(tmp_path)
    with NyplIndexLookup(out_path) as lookup:
        idf = _idf(lookup)
        config = _config()
        marc = MarcRecord(
            control_id="m",
            title="A study of widgets",
            title_main="A study of widgets",
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
            author_idf=idf,
            publisher_idf=idf,
            calibrator=None,
            combiner=WeightedMeanCombiner(config=config),
            pairings=compiled_pairings,
        )
    assert result.best is not None
    assert result.best.nypl_uuid == "UUID-0001"
    assert result.best.combined.raw > 70.0
    assert result.candidates_considered >= 1


def test_match_record_returns_no_best_when_below_floor(
    tmp_path: Path, compiled_pairings: CompiledPairings
) -> None:
    """A high min_combined_score floor filters all candidates out.

    The title shares the ``study`` token with UUID-0001 so the record is
    retrieved (candidates_considered >= 1), but the overall similarity is
    far below the 99.0 floor so no best match survives.
    """
    out_path = _build_tiny_index(tmp_path)
    with NyplIndexLookup(out_path) as lookup:
        idf = _idf(lookup)
        config = _config(min_score=99.0)
        marc = MarcRecord(
            control_id="m",
            title="A study of unrelated matters",
            title_main="A study of unrelated matters",
            publication_year=1940,
        )
        result = match_record(
            marc,
            lookup=lookup,
            config=config,
            idf=idf,
            author_idf=idf,
            publisher_idf=idf,
            calibrator=None,
            combiner=WeightedMeanCombiner(config=config),
            pairings=compiled_pairings,
        )
    assert result.best is None
    assert result.candidates_considered >= 1


def test_match_record_applies_calibrator(
    tmp_path: Path, compiled_pairings: CompiledPairings
) -> None:
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
            title_main="A study of widgets",
            publication_year=1940,
        )
        result = match_record(
            marc,
            lookup=lookup,
            config=config,
            idf=idf,
            author_idf=idf,
            publisher_idf=idf,
            calibrator=calibrator,
            combiner=WeightedMeanCombiner(config=config),
            pairings=compiled_pairings,
        )
    assert result.best is not None
    # Calibrated value should not equal raw/100 thanks to the supplied params.
    assert result.best.combined.calibrated != result.best.combined.raw / 100.0


def test_match_record_returns_alternates_in_descending_calibrated_order(
    tmp_path: Path, compiled_pairings: CompiledPairings
) -> None:
    """When multiple candidates pass the floor, alternates are sorted high→low."""
    out_path = _build_tiny_index(tmp_path)
    with NyplIndexLookup(out_path) as lookup:
        idf = _idf(lookup)
        config = _config(min_score=0.0)
        marc = MarcRecord(
            control_id="m",
            title="A study of widgets",
            title_main="A study of widgets",
            publication_year=1940,
        )
        result = match_record(
            marc,
            lookup=lookup,
            config=config,
            idf=idf,
            author_idf=idf,
            publisher_idf=idf,
            calibrator=None,
            combiner=WeightedMeanCombiner(config=config),
            pairings=compiled_pairings,
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
    tmp_path: Path, compiled_pairings: CompiledPairings
) -> None:
    """A series title that perfectly matches NYPL wins over a weaker primary.

    The primary title shares only the ``study`` token with UUID-0001 (enough
    to retrieve the candidate), while the series title matches it perfectly,
    so the series pairing must out-score the primary pairing.
    """
    out_path = _build_tiny_index(tmp_path)
    with NyplIndexLookup(out_path) as lookup:
        idf = _idf(lookup)
        config = _config(min_score=0.0)
        # marc.title is a weak partial; the series_titles contain the real match.
        marc = MarcRecord(
            control_id="m",
            title="A study of unrelated cover matters",
            title_main="A study of unrelated cover matters",
            series_titles=("A study of widgets",),
            publication_year=1940,
        )
        result = match_record(
            marc,
            lookup=lookup,
            config=config,
            idf=idf,
            author_idf=idf,
            publisher_idf=idf,
            calibrator=None,
            combiner=WeightedMeanCombiner(config=config),
            pairings=compiled_pairings,
        )
    assert result.best is not None
    title_evidence = next(ev for ev in result.best.evidence if ev.scorer == "title.token_set")
    losing_titles = [ev for ev in result.best.losing_evidence if ev.scorer == "title.token_set"]
    assert title_evidence.score > losing_titles[0].score


def test_match_record_includes_losing_evidence_from_alternate_pairings(
    tmp_path: Path, compiled_pairings: CompiledPairings
) -> None:
    """A MARC record with series titles records the losing title pairing."""
    out_path = _build_tiny_index(tmp_path)
    with NyplIndexLookup(out_path) as lookup:
        idf = _idf(lookup)
        config = _config(min_score=0.0)
        marc = MarcRecord(
            control_id="m",
            title="A study of widgets",
            title_main="A study of widgets",
            series_titles=("Some other series",),
            publication_year=1940,
        )
        result = match_record(
            marc,
            lookup=lookup,
            config=config,
            idf=idf,
            author_idf=idf,
            publisher_idf=idf,
            calibrator=None,
            combiner=WeightedMeanCombiner(config=config),
            pairings=compiled_pairings,
        )
    assert result.best is not None
    assert len(result.best.losing_evidence) >= 1


def _single_title_pairings() -> CompiledPairings:
    """Compile a config with only one title pairing and no author/publisher."""
    cfg = PairingConfig(
        marc_fields={"tm": FieldSpec(fields=("title_main",), combine="first")},
        cce_fields={"t": FieldSpec(fields=("title",), combine="first")},
        pairings=(PairingSpec(group="title", marc="tm", cce="t"),),
    )
    return compile_pairings(cfg)


def test_match_record_handles_groups_with_no_pairings(tmp_path: Path) -> None:
    """A config with empty author/publisher groups still scores the title."""
    out_path = _build_tiny_index(tmp_path)
    with NyplIndexLookup(out_path) as lookup:
        idf = _idf(lookup)
        config = _config(min_score=0.0)
        marc = MarcRecord(
            control_id="m",
            title="A study of widgets",
            title_main="A study of widgets",
            publication_year=1940,
        )
        result = match_record(
            marc,
            lookup=lookup,
            config=config,
            idf=idf,
            author_idf=idf,
            publisher_idf=idf,
            calibrator=None,
            combiner=WeightedMeanCombiner(config=config),
            pairings=_single_title_pairings(),
        )
    assert result.best is not None
    assert result.best.nypl_uuid == "UUID-0001"
    assert not any(ev.scorer == "name.author" for ev in result.best.evidence)
    assert not any(ev.scorer == "name.publisher" for ev in result.best.evidence)


def test_match_record_captures_winning_source_for_each_evidence(
    tmp_path: Path, compiled_pairings: CompiledPairings
) -> None:
    """``evidence_sources`` lines up 1:1 with ``evidence`` and labels each entry.

    Group scorers (title, author, publisher) carry the YAML pairing names of
    the winning pairing; non-group scorers carry the empty sentinel.
    """
    out_path = _build_tiny_index(tmp_path)
    with NyplIndexLookup(out_path) as lookup:
        idf = _idf(lookup)
        config = _config(min_score=0.0)
        marc = MarcRecord(
            control_id="m",
            title="A study of widgets",
            title_main="A study of widgets",
            main_author="Smith, John",
            publisher="Acme Press",
            publication_year=1940,
        )
        result = match_record(
            marc,
            lookup=lookup,
            config=config,
            idf=idf,
            author_idf=idf,
            publisher_idf=idf,
            calibrator=None,
            combiner=WeightedMeanCombiner(config=config),
            pairings=compiled_pairings,
        )
    assert result.best is not None
    assert len(result.best.evidence_sources) == len(result.best.evidence)
    by_scorer = dict(zip(result.best.evidence, result.best.evidence_sources, strict=True))
    title_source = next(src for ev, src in by_scorer.items() if ev.scorer == "title.token_set")
    assert title_source[0] in {"title_full", "title_main", "series_lead"}
    assert title_source[1] == "title"
    lccn_source = next(src for ev, src in by_scorer.items() if ev.scorer == "lccn.exact")
    assert lccn_source == ("", "")


def test_match_record_source_reflects_winning_pairing_when_cross_pairing_wins(
    tmp_path: Path,
) -> None:
    """When a group's second pairing wins, the captured source labels it.

    Compiles a one-group config with two title pairings where the second
    pairing (``backup_field`` ↔ ``title``) is the only one that can score
    non-zero because the first MARC field is absent. The captured source
    must point at the winning second pairing, not the first.
    """
    cfg = PairingConfig(
        marc_fields={
            "missing": FieldSpec(fields=("series_titles",), combine="first"),
            "backup": FieldSpec(fields=("title_main",), combine="first"),
        },
        cce_fields={"t": FieldSpec(fields=("title",), combine="first")},
        pairings=(
            PairingSpec(group="title", marc="missing", cce="t"),
            PairingSpec(group="title", marc="backup", cce="t"),
        ),
    )
    pairings = compile_pairings(cfg)
    out_path = _build_tiny_index(tmp_path)
    with NyplIndexLookup(out_path) as lookup:
        idf = _idf(lookup)
        config = _config(min_score=0.0)
        marc = MarcRecord(
            control_id="m",
            title="A study of widgets",
            title_main="A study of widgets",
            publication_year=1940,
        )
        result = match_record(
            marc,
            lookup=lookup,
            config=config,
            idf=idf,
            author_idf=idf,
            publisher_idf=idf,
            calibrator=None,
            combiner=WeightedMeanCombiner(config=config),
            pairings=pairings,
        )
    assert result.best is not None
    title_index = next(
        i for i, ev in enumerate(result.best.evidence) if ev.scorer == "title.token_set"
    )
    assert result.best.evidence_sources[title_index] == ("backup", "t")


def test_with_multiplier_preserves_evidence_fields_and_sets_weight() -> None:
    """``_with_multiplier`` returns a copy with the requested multiplier."""
    original = Evidence(
        scorer="name.author",
        score=42.0,
        max=100.0,
        skipped=False,
        decisive=False,
        features=(("k", 1.0),),
    )
    updated = _with_multiplier(original, 0.5)
    assert updated.scorer == original.scorer
    assert updated.score == original.score
    assert updated.max == original.max
    assert updated.skipped is original.skipped
    assert updated.decisive is original.decisive
    assert updated.features == original.features
    assert updated.weight_multiplier == 0.5


def test_match_record_downweights_author_for_translation_candidate(
    tmp_path: Path, compiled_pairings: CompiledPairings
) -> None:
    """A CCE entry whose desc flags translation gets author downweighted to 0.5.

    Builds a tiny one-record index with a ``<desc>`` containing
    ``"translated from the French"``. The translation signal fires, and the
    resulting author Evidence on the winning candidate carries
    ``weight_multiplier=0.5`` (set by the pipeline before combining).
    """
    reg_dir = tmp_path / "reg"
    ren_dir = tmp_path / "ren"
    reg_dir.mkdir()
    ren_dir.mkdir()
    (reg_dir / "tr_reg.xml").write_text(
        (
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            "<copyrightEntries>\n"
            '  <copyrightEntry regnum="A111111" id="UUID-TR-1">'
            "<author><authorName>Translator, Jane</authorName></author>"
            "<title>Translated work.</title>"
            '<regDate date="1955-01-01">Jan. 1, 1955</regDate>'
            '<publisher><pubName claimant="yes">House</pubName>'
            "<pubPlace>NYC</pubPlace></publisher>"
            "<desc>312 p. Translated from the French.</desc>"
            "</copyrightEntry>\n"
            "</copyrightEntries>\n"
        ),
        encoding="utf-8",
    )
    (ren_dir / "empty.tsv").write_text("", encoding="utf-8")
    out_path = tmp_path / "idx.lmdb"
    build_index(reg_dir=reg_dir, ren_dir=ren_dir, out_path=out_path)
    with NyplIndexLookup(out_path) as lookup:
        idf = _idf(lookup)
        config = _config(min_score=0.0)
        marc = MarcRecord(
            control_id="m",
            title="Translated work",
            title_main="Translated work",
            main_author="Original, Pierre",
            publication_year=1955,
        )
        result = match_record(
            marc,
            lookup=lookup,
            config=config,
            idf=idf,
            author_idf=idf,
            publisher_idf=idf,
            calibrator=None,
            combiner=WeightedMeanCombiner(config=config),
            pairings=compiled_pairings,
        )
    assert result.best is not None
    author_evidence = next(ev for ev in result.best.evidence if ev.scorer == "name.author")
    assert author_evidence.weight_multiplier == 0.5


def test_match_record_author_via_sor_pairing_recovers_match(
    tmp_path: Path, compiled_pairings: CompiledPairings
) -> None:
    """The sor↔author_name pairing supplies author signal main_author can't.

    UUID-0002 ("Le petit livre", 1955) records its author as
    "Dubois, David". This MARC record has no main author, so the default
    ``main_author↔author_name`` pairing is skipped; the statement of
    responsibility carries the name, and the ``sor↔author_name`` pairing
    is what produces a non-skipped author Evidence.
    """
    out_path = _build_tiny_index(tmp_path)
    with NyplIndexLookup(out_path) as lookup:
        idf = _idf(lookup)
        config = _config(min_score=0.0)
        marc = MarcRecord(
            control_id="m",
            title="Le petit livre",
            title_main="Le petit livre",
            main_author=None,
            statement_of_responsibility="David Dubois",
            publication_year=1955,
        )
        result = match_record(
            marc,
            lookup=lookup,
            config=config,
            idf=idf,
            author_idf=idf,
            publisher_idf=idf,
            calibrator=None,
            combiner=WeightedMeanCombiner(config=config),
            pairings=compiled_pairings,
        )
    assert result.best is not None
    assert result.best.nypl_uuid == "UUID-0002"
    author_evidence = next(ev for ev in result.best.evidence if ev.scorer == "name.author")
    assert author_evidence.skipped is False
    assert author_evidence.score > 0.0


def _ev(score: float, *, scorer: str = "test.scorer", skipped: bool = False) -> Evidence:
    return Evidence(
        scorer=scorer,
        score=score,
        max=100.0,
        skipped=skipped,
        decisive=False,
        features=(),
    )


def test_apply_title_isolation_multiplier_skips_when_title_index_out_of_range() -> None:
    """When no title Evidence was emitted, the helper is a no-op."""
    winning = [_ev(0.0, scorer="lccn.exact")]
    _apply_title_isolation_multiplier(winning, title_index=5)
    assert winning[0].weight_multiplier == 1.0


def test_apply_title_isolation_multiplier_skips_when_title_evidence_skipped() -> None:
    """A skipped title Evidence is left untouched."""
    winning = [
        _ev(0.0, scorer="title.token_set", skipped=True),
        _ev(0.0, scorer="name.author"),
    ]
    _apply_title_isolation_multiplier(winning, title_index=0)
    assert winning[0].weight_multiplier == 1.0


def test_apply_title_isolation_multiplier_fires_when_no_other_scorer_corroborates() -> None:
    winning = [
        _ev(100.0, scorer="title.token_set"),
        _ev(0.0, scorer="name.author"),
        _ev(0.0, scorer="year.exact"),
    ]
    _apply_title_isolation_multiplier(winning, title_index=0)
    assert winning[0].weight_multiplier == 0.3


def test_apply_title_isolation_multiplier_holds_when_one_other_scorer_corroborates() -> None:
    winning = [
        _ev(100.0, scorer="title.token_set"),
        _ev(75.0, scorer="name.author"),
        _ev(0.0, scorer="name.publisher"),
    ]
    _apply_title_isolation_multiplier(winning, title_index=0)
    assert winning[0].weight_multiplier == 1.0


def test_apply_title_isolation_multiplier_treats_year_as_non_corroborating() -> None:
    """Year alone does not corroborate: many records share a year."""
    winning = [
        _ev(100.0, scorer="title.token_set"),
        _ev(100.0, scorer="year.delta"),
        _ev(0.0, scorer="name.author"),
    ]
    _apply_title_isolation_multiplier(winning, title_index=0)
    assert winning[0].weight_multiplier == 0.3


def test_apply_title_isolation_multiplier_does_not_fire_when_title_score_is_low() -> None:
    """A weak title match is not downweighted — that would just raise the average."""
    winning = [
        _ev(20.0, scorer="title.token_set"),
        _ev(0.0, scorer="name.author"),
        _ev(0.0, scorer="name.publisher"),
    ]
    _apply_title_isolation_multiplier(winning, title_index=0)
    assert winning[0].weight_multiplier == 1.0


def test_match_record_downweights_title_when_no_other_scorer_corroborates(
    tmp_path: Path, compiled_pairings: CompiledPairings
) -> None:
    """A title-only match with no corroborating signal gets weight_multiplier 0.3.

    The CCE entry has a title that overlaps strongly with the MARC title, but
    every other field disagrees: different author, different year (outside the
    window), different publisher. No non-title scorer reaches the corroboration
    threshold (50.0), so the title's weight_multiplier drops to 0.3.
    """
    reg_dir = tmp_path / "reg"
    ren_dir = tmp_path / "ren"
    reg_dir.mkdir()
    ren_dir.mkdir()
    (reg_dir / "iso_reg.xml").write_text(
        (
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            "<copyrightEntries>\n"
            '  <copyrightEntry regnum="A222222" id="UUID-ISO-1">'
            "<author><authorName>Smith, John</authorName></author>"
            "<title>Selected widgets.</title>"
            '<regDate date="1920-01-01">Jan. 1, 1920</regDate>'
            '<publisher><pubName claimant="yes">Acme Press</pubName>'
            "<pubPlace>NYC</pubPlace></publisher>"
            "<desc>312 p.</desc>"
            "</copyrightEntry>\n"
            "</copyrightEntries>\n"
        ),
        encoding="utf-8",
    )
    (ren_dir / "empty.tsv").write_text("", encoding="utf-8")
    out_path = tmp_path / "idx.lmdb"
    build_index(reg_dir=reg_dir, ren_dir=ren_dir, out_path=out_path)
    with NyplIndexLookup(out_path) as lookup:
        idf = _idf(lookup)
        # Year window 5 lets the CCE candidate (1920) be retrieved for the MARC
        # (1925), but the 5-year delta drives the year scorer to zero, leaving
        # the title scorer alone with no corroborating signal.
        config = _config(min_score=0.0, year_window=5)
        marc = MarcRecord(
            control_id="m",
            title="Selected widgets",
            title_main="Selected widgets",
            main_author="Different, Name",
            publisher="Penguin Books",
            publication_year=1925,
        )
        result = match_record(
            marc,
            lookup=lookup,
            config=config,
            idf=idf,
            author_idf=idf,
            publisher_idf=idf,
            calibrator=None,
            combiner=WeightedMeanCombiner(config=config),
            pairings=compiled_pairings,
        )
    assert result.best is not None
    title_evidence = next(ev for ev in result.best.evidence if ev.scorer == "title.token_set")
    assert title_evidence.weight_multiplier == 0.3


def test_match_record_keeps_full_title_weight_when_author_corroborates(
    tmp_path: Path, compiled_pairings: CompiledPairings
) -> None:
    """A title match with author corroboration keeps full title weight.

    Author + title + year all match here, so corroboration is present and
    the title scorer's weight_multiplier stays at the default 1.0. Year
    alone would not corroborate (it is excluded from the corroboration
    set), but author does.
    """
    reg_dir = tmp_path / "reg"
    ren_dir = tmp_path / "ren"
    reg_dir.mkdir()
    ren_dir.mkdir()
    (reg_dir / "corr_reg.xml").write_text(
        (
            '<?xml version="1.0" encoding="UTF-8"?>\n'
            "<copyrightEntries>\n"
            '  <copyrightEntry regnum="A333333" id="UUID-CORR-1">'
            "<author><authorName>Smith, John</authorName></author>"
            "<title>Widget studies.</title>"
            '<regDate date="1955-01-01">Jan. 1, 1955</regDate>'
            '<publisher><pubName claimant="yes">House</pubName>'
            "<pubPlace>NYC</pubPlace></publisher>"
            "<desc>312 p.</desc>"
            "</copyrightEntry>\n"
            "</copyrightEntries>\n"
        ),
        encoding="utf-8",
    )
    (ren_dir / "empty.tsv").write_text("", encoding="utf-8")
    out_path = tmp_path / "idx.lmdb"
    build_index(reg_dir=reg_dir, ren_dir=ren_dir, out_path=out_path)
    with NyplIndexLookup(out_path) as lookup:
        idf = _idf(lookup)
        config = _config(min_score=0.0)
        marc = MarcRecord(
            control_id="m",
            title="Widget studies",
            title_main="Widget studies",
            main_author="Smith, John",
            publication_year=1955,
        )
        result = match_record(
            marc,
            lookup=lookup,
            config=config,
            idf=idf,
            author_idf=idf,
            publisher_idf=idf,
            calibrator=None,
            combiner=WeightedMeanCombiner(config=config),
            pairings=compiled_pairings,
        )
    assert result.best is not None
    title_evidence = next(ev for ev in result.best.evidence if ev.scorer == "title.token_set")
    assert title_evidence.weight_multiplier == 1.0
