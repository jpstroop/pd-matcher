"""Tests for :mod:`pd_matcher.match.scorers.renewal`."""

from pd_matcher.match.scorers.context import ScorerContext
from pd_matcher.match.scorers.renewal import CLAIMANT_CLASS_SCORER
from pd_matcher.match.scorers.renewal import NAME_CONDITIONED_SCORER
from pd_matcher.match.scorers.renewal import OREG_CLASS_SCORER
from pd_matcher.match.scorers.renewal import renewal_domain_evidence
from pd_matcher.match.scorers.renewal import score_claimant_class
from pd_matcher.match.scorers.renewal import score_claimant_name
from pd_matcher.match.scorers.renewal import score_oreg_class
from pd_matcher.models import MarcRecord
from pd_matcher.models import NyplRenRecord


def _marc(main_author: str | None = "Jane Albuquerque") -> MarcRecord:
    return MarcRecord(
        control_id="ctrl-1",
        title="A Title",
        title_main="A Title",
        main_author=main_author,
        language_code="eng",
    )


def _renewal(*, oreg: str | None = "A111111", claimants: str | None = None) -> NyplRenRecord:
    return NyplRenRecord(id="R1", entry_id="e1", oreg=oreg, claimants=claimants)


# --------------------------------------------------------------------------- #
# score_oreg_class
# --------------------------------------------------------------------------- #


def test_score_oreg_class_book_supports_match() -> None:
    ev = score_oreg_class(_renewal(oreg="A111111"))
    assert ev.scorer == OREG_CLASS_SCORER
    assert not ev.skipped
    assert ev.normalized == 1.0
    assert dict(ev.features)["is_book"] == 1.0


def test_score_oreg_class_periodical_argues_against_match() -> None:
    ev = score_oreg_class(_renewal(oreg="BB4567"))
    assert not ev.skipped
    assert ev.normalized == 0.0
    assert dict(ev.features)["is_periodical"] == 1.0


def test_score_oreg_class_drama_is_skipped_but_flagged() -> None:
    ev = score_oreg_class(_renewal(oreg="DP5"))
    assert ev.skipped
    assert dict(ev.features)["is_drama"] == 1.0


def test_score_oreg_class_other_class_is_skipped() -> None:
    ev = score_oreg_class(_renewal(oreg="E123"))
    assert ev.skipped


def test_score_oreg_class_absent_oreg_is_skipped() -> None:
    ev = score_oreg_class(_renewal(oreg=None))
    assert ev.skipped


# --------------------------------------------------------------------------- #
# score_claimant_class
# --------------------------------------------------------------------------- #


def test_score_claimant_class_author_full_support() -> None:
    ev = score_claimant_class(_renewal(claimants="Jane Doe|A"))
    assert ev.scorer == CLAIMANT_CLASS_SCORER
    assert not ev.skipped
    assert ev.normalized == 1.0


def test_score_claimant_class_estate_low_support() -> None:
    ev = score_claimant_class(_renewal(claimants="John Doe|W"))
    assert not ev.skipped
    assert ev.normalized == 0.15


def test_score_claimant_class_proprietor_mid_support() -> None:
    ev = score_claimant_class(_renewal(claimants="Acme Press|PWH"))
    assert not ev.skipped
    assert ev.normalized == 0.35


def test_score_claimant_class_author_wins_over_risk_class() -> None:
    ev = score_claimant_class(_renewal(claimants="Jane Doe|A||John Doe|W"))
    assert ev.normalized == 1.0


def test_score_claimant_class_estate_wins_over_proprietor() -> None:
    ev = score_claimant_class(_renewal(claimants="John Doe|W||Acme Press|PWH"))
    assert ev.normalized == 0.15


def test_score_claimant_class_no_recognized_code_is_skipped() -> None:
    ev = score_claimant_class(_renewal(claimants="Just A Name"))
    assert ev.skipped


# --------------------------------------------------------------------------- #
# score_claimant_name
# --------------------------------------------------------------------------- #


def test_score_claimant_name_author_uses_name_similarity(scorer_context: ScorerContext) -> None:
    renewal = _renewal(claimants="Jane Albuquerque|A")
    ev = score_claimant_name(_marc(), renewal, scorer_context)
    assert ev.scorer == NAME_CONDITIONED_SCORER
    assert not ev.skipped
    assert ev.normalized == 1.0


def test_score_claimant_name_estate_is_neutral(scorer_context: ScorerContext) -> None:
    renewal = _renewal(claimants="John Doe|W")
    ev = score_claimant_name(_marc(), renewal, scorer_context)
    assert ev.skipped


def test_score_claimant_name_proprietor_is_neutral(scorer_context: ScorerContext) -> None:
    renewal = _renewal(claimants="Acme Press|PWH")
    ev = score_claimant_name(_marc(), renewal, scorer_context)
    assert ev.skipped


def test_score_claimant_name_skipped_when_marc_author_absent(
    scorer_context: ScorerContext,
) -> None:
    renewal = _renewal(claimants="Jane Albuquerque|A")
    ev = score_claimant_name(_marc(main_author=None), renewal, scorer_context)
    assert ev.skipped
    assert dict(ev.features)["name_similarity"] == 0.0


# --------------------------------------------------------------------------- #
# renewal_domain_evidence
# --------------------------------------------------------------------------- #


def test_renewal_domain_evidence_returns_three_in_order(scorer_context: ScorerContext) -> None:
    renewal = _renewal(oreg="A111111", claimants="Jane Albuquerque|A")
    oreg_ev, claimant_ev, name_ev = renewal_domain_evidence(_marc(), renewal, scorer_context)
    assert oreg_ev.scorer == OREG_CLASS_SCORER
    assert claimant_ev.scorer == CLAIMANT_CLASS_SCORER
    assert name_ev.scorer == NAME_CONDITIONED_SCORER
