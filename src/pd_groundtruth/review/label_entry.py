"""Build the schema-7 vault entry the review server writes at label time.

When a labeler submits a verdict, the server writes a :class:`VaultEntry`
immediately. Because the three static CCE facts (``reg_year`` /
``renewal_year`` / ``was_renewed``) never change for a pair, they are stamped
onto that entry at label time rather than waiting for ``enrich-vault`` — a
newly-saved label is CCE-complete the moment it lands. The version-bound
``scores`` and ``matcher_version`` stay ``None`` here; those remain an
on-publish concern that ``enrich-vault`` recomputes against a specific build.

This is the pure, unit-tested core behind the server's verdict handler: it
takes the persisted :class:`ReviewPairRow` (which already carries the
denormalized CCE columns) plus the verdict inputs and returns the entry to
upsert. The renewal-year derivation is shared with ``enrich-vault`` via
:func:`pd_groundtruth.label_vault.renewal_year_of`, so there is one source of
that logic.
"""

from datetime import date

from msgspec.json import decode as json_decode

from pd_groundtruth.label_vault import SCHEMA_VERSION
from pd_groundtruth.label_vault import CategoryKey
from pd_groundtruth.label_vault import MatchSource
from pd_groundtruth.label_vault import VaultEntry
from pd_groundtruth.label_vault import extract_marc_identifiers
from pd_groundtruth.label_vault import renewal_year_of
from pd_groundtruth.review_db import PAIRING_RENEWAL
from pd_groundtruth.review_db import ReviewPairRow
from pd_matcher.models import MarcRecord


def _was_renewed(flag: int | None) -> bool | None:
    """Normalize the stored integer renewal flag to a tristate boolean.

    ``cce_was_renewed`` is persisted as ``1`` / ``0`` / ``None``; the vault's
    ``was_renewed`` is ``bool | None``. ``None`` is preserved so an unresolved
    source status stays distinct from a known "not renewed".
    """
    if flag is None:
        return None
    return bool(flag)


def _renewal_year(renewal_rdat: str | None) -> int | None:
    """Derive the renewal year from the row's ISO renewal-recording date.

    The row stores ``cce_renewal_rdat`` as an ISO-8601 string (or ``None``);
    parse it to a :class:`datetime.date` and defer to the shared
    :func:`renewal_year_of` so the rdat-to-year rule has one definition.
    """
    if renewal_rdat is None:
        return None
    return renewal_year_of(date.fromisoformat(renewal_rdat))


def _match_source(pairing_type: str) -> MatchSource:
    """Map a review pair's ``pairing_type`` to the vault's ``match_source``.

    A ``"renewal"`` pairing records ``"renewal"``; every other pairing
    (the default ``"registration"``) records ``"registration"``. The
    ``"both"`` value is reserved for a future pipeline that links a pair
    through both pathways and is never produced here.
    """
    return "renewal" if pairing_type == PAIRING_RENEWAL else "registration"


def build_label_entry(
    pair: ReviewPairRow,
    *,
    verdict: str,
    note: str | None,
    labeled_at: str,
    labeler: str,
    categories: tuple[CategoryKey, ...],
) -> VaultEntry:
    """Assemble the schema-7 vault entry for a freshly-submitted verdict.

    The human-entered fields (``verdict`` / ``note`` / ``categories`` /
    ``labeled_at`` / ``labeler``) and the CCE-side identifiers are taken
    verbatim; the three static CCE facts are derived off ``pair``'s
    denormalized columns. ``match_source`` is derived from the pair's
    ``pairing_type`` (a human-pathway fact preserved by ``enrich-vault``).
    ``scores`` and ``matcher_version`` are left ``None`` — they are
    version-bound and written only by ``enrich-vault``.

    Args:
        pair: The persisted review-pair row carrying ``marc_json`` and the
            denormalized CCE columns.
        verdict: The submitted verdict (``match`` / ``no_match`` / ``unsure``).
        note: The labeler's free-text rationale, or ``None``.
        labeled_at: The ISO-8601 timestamp shared with the DB label row.
        labeler: The labeler identifier.
        categories: The structured rationale tags for this verdict.
    """
    marc = json_decode(pair.marc_json.encode("utf-8"), type=MarcRecord)
    return VaultEntry(
        schema=SCHEMA_VERSION,
        marc_control_id=marc.control_id,
        nypl_uuid=pair.nypl_uuid,
        verdict=verdict,
        note=note,
        labeled_at=labeled_at,
        labeler=labeler,
        marc_identifiers=extract_marc_identifiers(marc),
        cce_regnum=pair.cce_regnum,
        cce_renewal_id=pair.cce_renewal_id,
        cce_renewal_oreg=pair.cce_renewal_oreg,
        categories=categories,
        reg_year=pair.cce_reg_year,
        renewal_year=_renewal_year(pair.cce_renewal_rdat),
        was_renewed=_was_renewed(pair.cce_was_renewed),
        scores=None,
        matcher_version=None,
        match_source=_match_source(pair.pairing_type),
    )


__all__ = [
    "build_label_entry",
]
