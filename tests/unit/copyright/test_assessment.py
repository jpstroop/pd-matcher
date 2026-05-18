"""Tests for :mod:`pd_matcher.copyright.assessment`."""

from msgspec import ValidationError
from msgspec import convert
from msgspec import to_builtins
from pytest import raises

from pd_matcher.copyright.assessment import CopyrightAssessment
from pd_matcher.copyright.status import CopyrightStatus


def test_assessment_is_frozen() -> None:
    """CopyrightAssessment is immutable after construction."""
    a = CopyrightAssessment(
        status=CopyrightStatus.PD_BY_AGE_PRE_95_YEARS,
        matched_rule_name="moving_wall_short_circuit",
        explanation="x",
        assumptions=(),
    )
    with raises(AttributeError):
        setattr(a, "matched_rule_name", "other")


def test_assessment_roundtrip() -> None:
    """The struct serializes to builtins and back without loss."""
    a = CopyrightAssessment(
        status=CopyrightStatus.UNKNOWN_NO_RULE_MATCHED,
        matched_rule_name=None,
        explanation="No rule matched",
        assumptions=("Assumption A", "Assumption B"),
    )
    again = convert(to_builtins(a), type=CopyrightAssessment)
    assert again == a


def test_assessment_forbids_unknown_fields() -> None:
    """Decoding rejects unknown fields."""
    with raises(ValidationError):
        convert(
            {
                "status": "PD_BY_AGE_PRE_95_YEARS",
                "matched_rule_name": None,
                "explanation": "x",
                "assumptions": [],
                "extra": 1,
            },
            type=CopyrightAssessment,
        )
