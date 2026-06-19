"""Tests for the cross-field publisher person/org gate (issue #86).

The publisher group pairs the MARC publisher / statement of responsibility
against the CCE ``author_name`` / ``claimants`` (and renewal twins) to catch
a corporate claimant that is in fact the publisher. The gate skips that
fallback when the CCE comparand is a person, so a person comparand neither
penalizes a real match (direction A) nor double-counts an author identity on
a no_match (direction B). The genuine ``publisher ↔ publisher_names`` pairing
is never gated.
"""

from pd_matcher.config.schemas import FieldSpec
from pd_matcher.match.evidence import Evidence
from pd_matcher.match.pairing_compiler import CceAccessor
from pd_matcher.match.pairing_compiler import CompiledPairing
from pd_matcher.match.pairing_compiler import _compile_cce_field
from pd_matcher.match.pipeline import _as_skipped
from pd_matcher.match.pipeline import _gate_cross_field_publisher
from pd_matcher.models import IndexedNyplRegRecord

_MAX: float = 100.0


def _cce_accessor(cce_name: str) -> CceAccessor:
    """Compile a single-raw-field CCE accessor exactly as the pipeline does."""
    spec = (
        FieldSpec(fields=(cce_name,), combine="join")
        if cce_name in {"claimants", "publisher_names"}
        else FieldSpec(fields=(cce_name,), combine="first")
    )
    return _compile_cce_field(cce_name, spec)


def _record(
    *,
    author_name: str | None = None,
    claimants: tuple[str, ...] = (),
    publisher_names: tuple[str, ...] = (),
    renewal_author: str | None = None,
    renewal_claimants: str | None = None,
) -> IndexedNyplRegRecord:
    return IndexedNyplRegRecord(
        uuid="u",
        title="t",
        was_renewed=False,
        author_name=author_name,
        claimants=claimants,
        publisher_names=publisher_names,
        renewal_author=renewal_author,
        renewal_claimants=renewal_claimants,
    )


def _pairing(cce_name: str, group: str = "publisher") -> CompiledPairing:
    return CompiledPairing(
        group=group,
        marc_name="publisher",
        cce_name=cce_name,
        marc_accessor=lambda marc: None,
        cce_accessor=_cce_accessor(cce_name),
    )


def _scored(score: float, *, skipped: bool = False) -> Evidence:
    return Evidence(
        scorer="name.publisher",
        score=score,
        max=_MAX,
        skipped=skipped,
        decisive=False,
        features=(("token_overlap", 1.0),),
    )


def test_person_author_cross_field_is_skipped() -> None:
    """publisher ↔ author_name on a person comparand drops out (direction A)."""
    pairing = _pairing("author_name")
    record = _record(author_name="Britcher, Phyllis I.")
    gated = _gate_cross_field_publisher(pairing, record, _scored(0.0))
    assert gated.skipped is True
    assert gated.score == 0.0


def test_person_claimant_cross_field_is_skipped() -> None:
    """publisher ↔ claimants on a person comparand drops out (direction B)."""
    pairing = _pairing("claimants")
    record = _record(claimants=("Maxwell Geismar",))
    gated = _gate_cross_field_publisher(pairing, record, _scored(100.0))
    assert gated.skipped is True
    assert gated.score == 0.0


def test_org_claimant_cross_field_is_kept() -> None:
    """A corporate claimant comparand still feeds the publisher signal."""
    pairing = _pairing("claimants")
    record = _record(claimants=("Judson Press",))
    evidence = _scored(95.0)
    gated = _gate_cross_field_publisher(pairing, record, evidence)
    assert gated is evidence
    assert gated.skipped is False


def test_org_author_cross_field_is_kept() -> None:
    """A corporate author_name comparand still feeds the publisher signal."""
    pairing = _pairing("author_name")
    record = _record(author_name="Cornell University")
    evidence = _scored(80.0)
    gated = _gate_cross_field_publisher(pairing, record, evidence)
    assert gated is evidence


def test_mixed_claimant_tuple_with_org_is_kept() -> None:
    """A claimants tuple containing any org marker is treated as an org."""
    pairing = _pairing("claimants")
    record = _record(claimants=("University Microfilms Library Services", "Frank Bannister"))
    evidence = _scored(70.0)
    assert _gate_cross_field_publisher(pairing, record, evidence) is evidence


def test_person_renewal_author_cross_field_is_skipped() -> None:
    pairing = _pairing("renewal_author")
    record = _record(renewal_author="Howland, Arthur Hoag")
    assert _gate_cross_field_publisher(pairing, record, _scored(0.0)).skipped is True


def test_person_renewal_claimants_cross_field_is_skipped() -> None:
    pairing = _pairing("renewal_claimants")
    record = _record(renewal_claimants="Gerald S. Snyder")
    assert _gate_cross_field_publisher(pairing, record, _scored(0.0)).skipped is True


def test_genuine_publisher_names_pairing_is_never_gated() -> None:
    """publisher ↔ publisher_names is the real comparison; never gated."""
    pairing = _pairing("publisher_names")
    record = _record(publisher_names=("Gosset & Dunlap",))
    evidence = _scored(0.0)
    assert _gate_cross_field_publisher(pairing, record, evidence) is evidence


def test_non_publisher_group_is_never_gated() -> None:
    """The author group passes through untouched even on person comparands."""
    pairing = _pairing("author_name", group="author")
    record = _record(author_name="Britcher, Phyllis I.")
    evidence = _scored(100.0)
    assert _gate_cross_field_publisher(pairing, record, evidence) is evidence


def test_already_skipped_evidence_passes_through() -> None:
    """An already-skipped cross-field Evidence is returned unchanged."""
    pairing = _pairing("author_name")
    record = _record(author_name="Britcher, Phyllis I.")
    evidence = _scored(0.0, skipped=True)
    assert _gate_cross_field_publisher(pairing, record, evidence) is evidence


def test_as_skipped_zeroes_score_and_clears_features() -> None:
    skipped = _as_skipped(_scored(73.0))
    assert skipped.skipped is True
    assert skipped.score == 0.0
    assert skipped.features == ()
    assert skipped.scorer == "name.publisher"
