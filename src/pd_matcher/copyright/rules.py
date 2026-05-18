"""Ordered rule engine that evaluates :class:`Facts` against the Cornell matrix.

The engine resolves each YAML :class:`PredicateCall` to a callable from
:mod:`pd_matcher.copyright.predicates` (returns ``bool``) or
:mod:`pd_matcher.copyright.inference` (returns ``(bool, str | None)``).
Both are wrapped into a single unified signature
``Callable[[Facts, tuple[int | float, ...]], tuple[bool, str | None]]``
so the engine does not need to switch on predicate kind at evaluation
time. The moving-wall short-circuit is evaluated *before* any YAML rule
so the cheapest legally-certain status surfaces first.
"""

from collections.abc import Callable
from collections.abc import Iterable

from pd_matcher.config.schemas import CopyrightRule
from pd_matcher.config.schemas import CopyrightRuleSet
from pd_matcher.config.schemas import PredicateCall
from pd_matcher.copyright.assessment import CopyrightAssessment
from pd_matcher.copyright.facts import Facts
from pd_matcher.copyright.inference import foreign_in_pd_home_country_1996
from pd_matcher.copyright.inference import has_us_notice
from pd_matcher.copyright.inference import is_us_government_work
from pd_matcher.copyright.predicates import country_delayed_uraa
from pd_matcher.copyright.predicates import country_is_foreign
from pd_matcher.copyright.predicates import country_is_us
from pd_matcher.copyright.predicates import country_no_treaty
from pd_matcher.copyright.predicates import in_pd_by_age
from pd_matcher.copyright.predicates import match_confidence_at_least
from pd_matcher.copyright.predicates import published_before
from pd_matcher.copyright.predicates import published_between
from pd_matcher.copyright.predicates import published_on_or_after
from pd_matcher.copyright.predicates import was_registered
from pd_matcher.copyright.predicates import was_renewed
from pd_matcher.copyright.status import CopyrightStatus

UnifiedPredicate = Callable[[Facts, tuple[int | float, ...]], tuple[bool, str | None]]


class RuleEvaluationError(Exception):
    """Raised when a rule references an unknown predicate or wrong arg count."""


def _wrap_zero_arg(fn: Callable[[Facts], bool]) -> UnifiedPredicate:
    """Adapt a zero-argument :mod:`predicates` callable to the unified shape."""

    def call(facts: Facts, args: tuple[int | float, ...]) -> tuple[bool, str | None]:
        if args:
            raise RuleEvaluationError(
                f"predicate {fn.__name__!r} expects no args (got {len(args)})"
            )
        return fn(facts), None

    call.__name__ = fn.__name__
    return call


def _wrap_int_int(fn: Callable[[Facts, int, int], bool]) -> UnifiedPredicate:
    """Adapt an ``(int, int)`` :mod:`predicates` callable to the unified shape."""

    def call(facts: Facts, args: tuple[int | float, ...]) -> tuple[bool, str | None]:
        if len(args) != 2:
            raise RuleEvaluationError(f"predicate {fn.__name__!r} expects 2 args (got {len(args)})")
        lo, hi = args
        return fn(facts, int(lo), int(hi)), None

    call.__name__ = fn.__name__
    return call


def _wrap_int(fn: Callable[[Facts, int], bool]) -> UnifiedPredicate:
    """Adapt a single-``int`` :mod:`predicates` callable to the unified shape."""

    def call(facts: Facts, args: tuple[int | float, ...]) -> tuple[bool, str | None]:
        if len(args) != 1:
            raise RuleEvaluationError(f"predicate {fn.__name__!r} expects 1 arg (got {len(args)})")
        (year,) = args
        return fn(facts, int(year)), None

    call.__name__ = fn.__name__
    return call


def _wrap_float(fn: Callable[[Facts, float], bool]) -> UnifiedPredicate:
    """Adapt a single-``float`` :mod:`predicates` callable to the unified shape."""

    def call(facts: Facts, args: tuple[int | float, ...]) -> tuple[bool, str | None]:
        if len(args) != 1:
            raise RuleEvaluationError(f"predicate {fn.__name__!r} expects 1 arg (got {len(args)})")
        (threshold,) = args
        return fn(facts, float(threshold)), None

    call.__name__ = fn.__name__
    return call


def _wrap_inference(
    fn: Callable[[Facts], tuple[bool, str | None]],
) -> UnifiedPredicate:
    """Adapt an :mod:`inference` callable to the unified shape (no args allowed)."""

    def call(facts: Facts, args: tuple[int | float, ...]) -> tuple[bool, str | None]:
        if args:
            raise RuleEvaluationError(
                f"inference {fn.__name__!r} expects no args (got {len(args)})"
            )
        return fn(facts)

    call.__name__ = fn.__name__
    return call


_REGISTRY: dict[str, UnifiedPredicate] = {
    "in_pd_by_age": _wrap_zero_arg(in_pd_by_age),
    "country_is_us": _wrap_zero_arg(country_is_us),
    "country_is_foreign": _wrap_zero_arg(country_is_foreign),
    "country_no_treaty": _wrap_zero_arg(country_no_treaty),
    "country_delayed_uraa": _wrap_zero_arg(country_delayed_uraa),
    "was_registered": _wrap_zero_arg(was_registered),
    "was_renewed": _wrap_zero_arg(was_renewed),
    "published_between": _wrap_int_int(published_between),
    "published_before": _wrap_int(published_before),
    "published_on_or_after": _wrap_int(published_on_or_after),
    "match_confidence_at_least": _wrap_float(match_confidence_at_least),
    "has_us_notice": _wrap_inference(has_us_notice),
    "is_us_government_work": _wrap_inference(is_us_government_work),
    "foreign_in_pd_home_country_1996": _wrap_inference(foreign_in_pd_home_country_1996),
}


def _resolve_status(name: str, rule_name: str) -> CopyrightStatus:
    """Resolve a ``then:`` string to a :class:`CopyrightStatus` or raise."""
    try:
        return CopyrightStatus[name]
    except KeyError as exc:
        raise RuleEvaluationError(f"rule {rule_name!r}: unknown CopyrightStatus {name!r}") from exc


def _evaluate_call(
    call: PredicateCall,
    facts: Facts,
    *,
    enable_assumptions: bool,
) -> tuple[bool, str | None]:
    """Resolve and evaluate one :class:`PredicateCall`, returning ``(value, assumption)``."""
    fn = _REGISTRY.get(call.predicate)
    if fn is None:
        raise RuleEvaluationError(f"unknown predicate {call.predicate!r}")
    value, assumption = fn(facts, call.args)
    if call.negate:
        value = not value
    if assumption is not None and not enable_assumptions:
        return False, None
    return value, assumption


def _evaluate_rule(
    rule: CopyrightRule,
    facts: Facts,
    *,
    enable_assumptions: bool,
) -> tuple[bool, tuple[str, ...]]:
    """Evaluate every ``when`` predicate in order.

    Returns:
        ``(matched, assumptions)`` where ``matched`` is ``True`` only when
        every predicate returned ``True`` after negation, and
        ``assumptions`` is the ordered tuple of dynamic assumptions
        surfaced by inference predicates that contributed to the match.
    """
    assumptions: list[str] = []
    for call in rule.when:
        value, assumption = _evaluate_call(call, facts, enable_assumptions=enable_assumptions)
        if not value:
            return False, ()
        if assumption is not None:
            assumptions.append(assumption)
    return True, tuple(assumptions)


def assess(
    facts: Facts,
    ruleset: CopyrightRuleSet,
    *,
    enable_assumptions: bool = True,
) -> CopyrightAssessment:
    """Return a :class:`CopyrightAssessment` for ``facts`` against ``ruleset``.

    Args:
        facts: The structured input over which predicates evaluate.
        ruleset: An ordered list of :class:`CopyrightRule`; first match
            wins.
        enable_assumptions: When ``False`` any inference predicate that
            would contribute a documented assumption is treated as
            ``False`` so the rule it gates cannot fire. Defaults to
            ``True``.

    Returns:
        A :class:`CopyrightAssessment` describing the verdict.
    """
    if in_pd_by_age(facts):
        return CopyrightAssessment(
            status=CopyrightStatus.PD_BY_AGE_PRE_95_YEARS,
            matched_rule_name="moving_wall_short_circuit",
            explanation=(
                f"Published in {facts.pub_year}; more than 95 years before "
                f"{facts.today.isoformat()}."
            ),
            assumptions=(),
        )
    for rule in ruleset.rules:
        matched, dynamic_assumptions = _evaluate_rule(
            rule, facts, enable_assumptions=enable_assumptions
        )
        if matched:
            return CopyrightAssessment(
                status=_resolve_status(rule.then, rule.name),
                matched_rule_name=rule.name,
                explanation=rule.explanation,
                assumptions=tuple(rule.assumptions) + dynamic_assumptions,
            )
    return CopyrightAssessment(
        status=CopyrightStatus.UNKNOWN_NO_RULE_MATCHED,
        matched_rule_name=None,
        explanation="No rule matched the observed facts.",
        assumptions=(),
    )


def registered_predicate_names() -> Iterable[str]:
    """Return the names of every predicate currently registered."""
    return _REGISTRY.keys()


__all__ = [
    "RuleEvaluationError",
    "assess",
    "registered_predicate_names",
]
