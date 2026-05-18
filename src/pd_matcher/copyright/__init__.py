"""Copyright rule engine — Phase 5 of the pd_matcher pipeline.

Public entry point: :func:`assess_record`. Given a parsed MARC record and
the matcher's verdict against the CCE registration corpus (the U.S.
Copyright Office's Catalog of Copyright Entries, published by the
Library of Congress and transcribed into XML/TSV by NYPL), it returns a
:class:`CopyrightAssessment` describing the work's public-domain status,
the rule that fired, and any documented assumptions the inference layer
relied on.

The shipped Cornell ruleset is loaded once at import time from
``pd_matcher/config/defaults/copyright_rules.yaml`` and cached for the
process lifetime. Tests and the (Phase 7) CLI ``--as-of`` flag may
override the reference date.
"""

from datetime import date
from importlib.resources import as_file
from importlib.resources import files

from pd_matcher.config.loader import load_copyright_rules
from pd_matcher.config.schemas import CopyrightRuleSet
from pd_matcher.copyright.assessment import CopyrightAssessment
from pd_matcher.copyright.facts import Facts
from pd_matcher.copyright.facts import build_facts
from pd_matcher.copyright.rules import RuleEvaluationError
from pd_matcher.copyright.rules import assess
from pd_matcher.copyright.status import CopyrightStatus
from pd_matcher.match.result import MatchResult
from pd_matcher.models import IndexedNyplRegRecord
from pd_matcher.models import MarcRecord


def _load_default_ruleset() -> CopyrightRuleSet:
    """Read the packaged ``copyright_rules.yaml`` and return the rule set."""
    resource = files("pd_matcher.config.defaults") / "copyright_rules.yaml"
    with as_file(resource) as path:
        return load_copyright_rules(path)


_DEFAULT_RULESET: CopyrightRuleSet = _load_default_ruleset()


def default_ruleset() -> CopyrightRuleSet:
    """Return the cached shipped Cornell ruleset."""
    return _DEFAULT_RULESET


def assess_record(
    marc: MarcRecord,
    match: MatchResult | None,
    *,
    today: date | None = None,
    matched_nypl: IndexedNyplRegRecord | None = None,
    ruleset: CopyrightRuleSet | None = None,
    enable_assumptions: bool = True,
) -> CopyrightAssessment:
    """Return a :class:`CopyrightAssessment` for one MARC record.

    Args:
        marc: The MARC bibliographic record under evaluation.
        match: The matcher's verdict, or ``None`` when no matching pass
            has been run.
        today: Reference date for age-sensitive predicates; defaults to
            :meth:`date.today`. Pin a value for tests and reproducible
            runs.
        matched_nypl: The hydrated CCE registration corresponding to
            ``match.best`` (loaded from the NYPL-transcribed index).
            Optional; supply it to enable publisher-based inference
            (e.g. US-government detection) on the registration side.
        ruleset: An override ruleset; defaults to the shipped Cornell
            matrix.
        enable_assumptions: When ``False``, predicates that surface a
            documented assumption are treated as ``False`` so they
            cannot gate a rule.

    Returns:
        A frozen :class:`CopyrightAssessment`.
    """
    reference_date = today if today is not None else date.today()
    facts = build_facts(
        marc,
        match,
        today=reference_date,
        matched_nypl=matched_nypl,
    )
    active_ruleset = ruleset if ruleset is not None else _DEFAULT_RULESET
    return assess(facts, active_ruleset, enable_assumptions=enable_assumptions)


__all__ = [
    "CopyrightAssessment",
    "CopyrightStatus",
    "Facts",
    "RuleEvaluationError",
    "assess",
    "assess_record",
    "build_facts",
    "default_ruleset",
]
