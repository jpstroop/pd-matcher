"""Publisher==claimant routing decision for the name scorer groups (issue #86).

The NYPL transcription echoes the copyright claimant into the CCE publisher
slot near-universally — ``publisher == claimant`` fires on 93.4% of all
registrations. That echo makes the stock publisher pairing compare a MARC
publisher against a *person's name* (the claimant, who is usually the author),
fabricating a ``name.publisher = 0.0`` that drags a genuine match down
(direction A) or, on a same-author-different-work pair, a spurious
``name.publisher = 1.0`` that inflates a non-match (direction B).

The routing rule resolves both with one mechanism. When a normalized value
appears in **both** ``publisher_names`` and ``claimants``, that shared value is
scored once against the best of the MARC author fields and the best of the MARC
publisher fields. When the winning score clears
:attr:`~pd_matcher.config.schemas.MatchingConfig.claimant_routing_floor`, the
value is *routed* to the winning field: it becomes that field's evidence and the
losing field's comparison drops out of the combiner as missing (never a
mismatch). A single distinct value therefore contributes evidence at most once.
When the winning score is below the floor the value matches neither MARC field,
so re-routing would merely discard a legitimate publisher disagreement; stock
evidence is left untouched so a real mismatch survives.

This module holds the *decision* (which shared keys route where); the pipeline
applies it to the per-pairing evidence.
"""

from msgspec import Struct

from pd_matcher.match.scorers.context import ScorerContext
from pd_matcher.match.scorers.name import score_author
from pd_matcher.match.scorers.name import score_publisher
from pd_matcher.models import MarcRecord
from pd_matcher.normalize.text import tokenize

AUTHOR_SCORER: str = "name.author"
PUBLISHER_SCORER: str = "name.publisher"

_GENUINE_PUBLISHER_CCE: str = "publisher_names"


class RoutingDecision(Struct, frozen=True, forbid_unknown_fields=True):
    """Which shared ``publisher==claimant`` keys route to which name group.

    ``author_routed`` holds the normalized token-set keys whose shared value
    was routed to the author group — the publisher group must drop any pairing
    won by one of these keys. ``publisher_routed`` is the symmetric set for the
    author group. A key never appears in both sets (single-contribution guard).
    An all-empty decision means the rule did not fire for this pair.
    """

    author_routed: frozenset[frozenset[str]]
    publisher_routed: frozenset[frozenset[str]]

    @property
    def fired(self) -> bool:
        """Return ``True`` when at least one shared value was routed."""
        return bool(self.author_routed) or bool(self.publisher_routed)


def value_key(value: str) -> frozenset[str]:
    """Return the normalized token-set key used to compare CCE values."""
    return frozenset(tokenize(value))


def _key_index(values: tuple[str, ...]) -> dict[frozenset[str], str]:
    """Return a ``token-set key -> original value`` map, dropping empty keys."""
    index: dict[frozenset[str], str] = {}
    for value in values:
        if not value:
            continue
        key = value_key(value)
        if key:
            index[key] = value
    return index


def is_blank_publisher(publisher_names: tuple[str, ...]) -> bool:
    """Return ``True`` when the CCE record carries no usable publisher name."""
    return not any(publisher_names)


def compute_routing(
    marc: MarcRecord,
    publisher_names: tuple[str, ...],
    claimants: tuple[str, ...],
    ctx: ScorerContext,
) -> RoutingDecision:
    """Decide which ``publisher==claimant`` shared values route where.

    For every normalized value present in both ``publisher_names`` and
    ``claimants``, the value is scored against the best of the MARC author
    fields (``main_author``, ``statement_of_responsibility``) and the best of
    the MARC publisher fields (``publisher``, ``statement_of_responsibility``).
    The value routes to the higher-scoring field only when that score clears
    :attr:`MatchingConfig.claimant_routing_floor`; a tie routes to the author,
    matching the pipeline's stable-first argmax. A sub-floor winner leaves the
    value unrouted so a genuine publisher mismatch is preserved.

    Args:
        marc: The MARC record under comparison.
        publisher_names: The candidate's CCE ``publisher_names`` tuple.
        claimants: The candidate's CCE ``claimants`` tuple.
        ctx: The per-record scorer context (supplies the floor and IDF tables).

    Returns:
        A :class:`RoutingDecision`; ``fired`` is ``False`` when nothing routed.
    """
    publisher_keys = _key_index(publisher_names)
    claimant_keys = _key_index(claimants)
    shared = publisher_keys.keys() & claimant_keys.keys()
    if not shared:
        return RoutingDecision(author_routed=frozenset(), publisher_routed=frozenset())
    floor = ctx.config.claimant_routing_floor
    marc_author_fields = (marc.main_author, marc.statement_of_responsibility)
    marc_publisher_fields = (marc.publisher, marc.statement_of_responsibility)
    author_routed: set[frozenset[str]] = set()
    publisher_routed: set[frozenset[str]] = set()
    for key in shared:
        value = claimant_keys[key]
        author_score = max(
            (score_author(field, value, ctx).normalized for field in marc_author_fields if field),
            default=0.0,
        )
        publisher_score = max(
            (
                score_publisher(field, value, ctx).normalized
                for field in marc_publisher_fields
                if field
            ),
            default=0.0,
        )
        if max(author_score, publisher_score) < floor:
            continue
        if author_score >= publisher_score:
            author_routed.add(key)
        else:
            publisher_routed.add(key)
    return RoutingDecision(
        author_routed=frozenset(author_routed),
        publisher_routed=frozenset(publisher_routed),
    )


__all__ = [
    "AUTHOR_SCORER",
    "PUBLISHER_SCORER",
    "RoutingDecision",
    "compute_routing",
    "is_blank_publisher",
    "value_key",
]
