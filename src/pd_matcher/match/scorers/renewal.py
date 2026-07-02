"""Domain scorers for the three validated MARC↔renewal signals (issue #45).

Eval v2 (``docs/findings/renewal_matcher_eval_2026-07-01.md``) validated three
features read straight off the renewal record, each with a correctly-signed
coefficient in a trained model:

1. **``oreg`` class** — the original-registration class token
   (:func:`pd_matcher.normalize.registration_numbers.reg_class`). A book-family
   original supports a monograph match (coef +1.16); a periodical original
   argues against it (coef -0.92).
2. **claimant class** — the statutory renewal-right class parsed from the
   claimants (:func:`pd_matcher.normalize.claimants.parse_claimants`). An estate
   (coef -0.49) or proprietor (coef -0.31) renewal is a mismatch risk relative
   to an author renewal.
3. **class-conditioned author name-match** — the claimant-vs-MARC-author name
   similarity, meaningful *only* when the renewal is claimed as author (coef
   +0.30). An estate or proprietor renews under a different name, so the signal
   is skipped (neutral, never a penalty) unless an author-coded claimant exists.

Each scorer emits an :class:`pd_matcher.match.evidence.Evidence` whose
``normalized`` reading is a ``[0, 1]`` match-support value, so the weighted-mean
combiner can average them alongside the title / author / claimant / year
Evidence and a learned combiner can read them as features. A signal that does
not apply to a pair is ``skipped`` — excluded from the mean and neutral, exactly
as an absent field scorer is.
"""

from pd_matcher.match.evidence import Evidence
from pd_matcher.match.scorers.context import ScorerContext
from pd_matcher.match.scorers.name import score_author
from pd_matcher.models import MarcRecord
from pd_matcher.models import NyplRenRecord
from pd_matcher.normalize.claimants import author_claimant_name
from pd_matcher.normalize.claimants import claimant_class_indicators
from pd_matcher.normalize.claimants import parse_claimants
from pd_matcher.normalize.registration_numbers import reg_class

OREG_CLASS_SCORER: str = "renewal.oreg_class"
CLAIMANT_CLASS_SCORER: str = "renewal.claimant_class"
NAME_CONDITIONED_SCORER: str = "renewal.name_conditioned"

_MAX: float = 1.0

_OREG_BOOK: frozenset[str] = frozenset({"A", "AA", "AF", "AI", "AFO", "AIO"})
_OREG_PERIODICAL: frozenset[str] = frozenset({"B", "BB"})
_OREG_DRAMA: frozenset[str] = frozenset({"D", "DF", "DP"})

_BOOK_SUPPORT: float = 1.0
_PERIODICAL_SUPPORT: float = 0.0

_AUTHOR_SUPPORT: float = 1.0
_ESTATE_SUPPORT: float = 0.15
_PROPRIETOR_SUPPORT: float = 0.35


def score_oreg_class(renewal: NyplRenRecord) -> Evidence:
    """Return :class:`Evidence` for the renewal's original-registration class.

    A book-family original registration supports a monograph match (normalized
    ``1.0``); a periodical original argues against it (normalized ``0.0``). Any
    other class — drama, an unrecognized class, or an absent ``oreg`` — is
    ``skipped`` (neutral). The book / periodical / drama membership is recorded
    as sub-features for the learned combiner.
    """
    cls = reg_class(renewal.oreg)
    is_book = cls in _OREG_BOOK
    is_periodical = cls in _OREG_PERIODICAL
    features: tuple[tuple[str, float], ...] = (
        ("is_book", 1.0 if is_book else 0.0),
        ("is_periodical", 1.0 if is_periodical else 0.0),
        ("is_drama", 1.0 if cls in _OREG_DRAMA else 0.0),
    )
    if is_book:
        return Evidence(OREG_CLASS_SCORER, _BOOK_SUPPORT, _MAX, False, False, features)
    if is_periodical:
        return Evidence(OREG_CLASS_SCORER, _PERIODICAL_SUPPORT, _MAX, False, False, features)
    return Evidence(OREG_CLASS_SCORER, 0.0, _MAX, True, False, features)


def score_claimant_class(renewal: NyplRenRecord) -> Evidence:
    """Return :class:`Evidence` for the renewal's statutory claimant class.

    An author claim is full match support (normalized ``1.0``); an estate claim
    is a stronger mismatch risk than a proprietor claim, so it supports less
    (``0.15`` vs ``0.35``), matching the trained coefficients (estate -0.49,
    proprietor -0.31). When an author claim co-occurs with a risk class the
    author reading wins; a renewal with no recognizable claimant class is
    ``skipped`` (neutral). The per-class indicators are recorded as
    sub-features.
    """
    pairs = parse_claimants(renewal.claimants)
    is_author, is_estate, is_proprietor = claimant_class_indicators(pairs)
    features: tuple[tuple[str, float], ...] = (
        ("is_author", 1.0 if is_author else 0.0),
        ("is_estate", 1.0 if is_estate else 0.0),
        ("is_proprietor", 1.0 if is_proprietor else 0.0),
    )
    if is_author:
        support = _AUTHOR_SUPPORT
    elif is_estate:
        support = _ESTATE_SUPPORT
    elif is_proprietor:
        support = _PROPRIETOR_SUPPORT
    else:
        return Evidence(CLAIMANT_CLASS_SCORER, 0.0, _MAX, True, False, features)
    return Evidence(CLAIMANT_CLASS_SCORER, support, _MAX, False, False, features)


def score_claimant_name(marc: MarcRecord, renewal: NyplRenRecord, ctx: ScorerContext) -> Evidence:
    """Return :class:`Evidence` for the class-conditioned author name-match.

    The claimant-vs-MARC-author name similarity is only meaningful when the
    renewal is claimed *as author*: an estate or proprietor renews under a
    different name, so the signal is ``skipped`` (neutral, never a penalty)
    unless an author-coded claimant exists. When one does, the first author
    claimant's name is scored against the MARC author with the production author
    scorer and its normalized reading is carried through; a skip on the
    underlying author scorer (missing MARC author) propagates as a skip here.
    """
    name = author_claimant_name(parse_claimants(renewal.claimants))
    if name is None:
        return Evidence(NAME_CONDITIONED_SCORER, 0.0, _MAX, True, False, ())
    inner = score_author(marc.main_author, name, ctx)
    features: tuple[tuple[str, float], ...] = (("name_similarity", inner.normalized),)
    if inner.skipped:
        return Evidence(NAME_CONDITIONED_SCORER, 0.0, _MAX, True, False, features)
    return Evidence(NAME_CONDITIONED_SCORER, inner.normalized, _MAX, False, False, features)


def renewal_domain_evidence(
    marc: MarcRecord, renewal: NyplRenRecord, ctx: ScorerContext
) -> tuple[Evidence, Evidence, Evidence]:
    """Return the three renewal-domain Evidence for one MARC↔renewal pairing.

    The tuple is ``(oreg_class, claimant_class, name_conditioned)`` — the order
    the renewal scoring path folds into its evidence sequence and payload.
    """
    return (
        score_oreg_class(renewal),
        score_claimant_class(renewal),
        score_claimant_name(marc, renewal, ctx),
    )


__all__ = [
    "CLAIMANT_CLASS_SCORER",
    "NAME_CONDITIONED_SCORER",
    "OREG_CLASS_SCORER",
    "renewal_domain_evidence",
    "score_claimant_class",
    "score_claimant_name",
    "score_oreg_class",
]
