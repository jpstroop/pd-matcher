"""Typed ``sqlite3`` wrapper for the self-contained review database.

Phase 2a writes proposed ``(MARC, CCE-candidate)`` pairs into a single
SQLite file that Phase 2b's UI reads to present rich review cards and to
record human verdicts. This is a *review queue*, not a set of confirmed
matches: a row's presence only means the matcher proposed the pair and the
stratified sampler selected it for labeling.

The schema is two tables. ``review_pair`` holds one row per proposed pair
plus denormalized convenience columns (``marc_*`` / ``cce_*``) and the
lossless ``marc_json`` blob (the full parsed :class:`MarcRecord`). ``label``
holds verdicts in an append-only log keyed on ``pair_id`` so re-labels keep
history; the "current" label is the latest by ``(labeled_at, id)``.
"""

from collections.abc import Iterator
from datetime import UTC
from datetime import datetime
from pathlib import Path
from sqlite3 import Connection
from sqlite3 import Row
from sqlite3 import connect as sqlite_connect

from msgspec import Struct
from msgspec.json import decode as json_decode
from msgspec.json import encode as json_encode

from pd_groundtruth.label_vault import CategoryKey

VERDICT_MATCH: str = "match"
VERDICT_NO_MATCH: str = "no_match"
VERDICT_UNSURE: str = "unsure"

_VALID_VERDICTS: frozenset[str] = frozenset({VERDICT_MATCH, VERDICT_NO_MATCH, VERDICT_UNSURE})

PAIRING_REGISTRATION: str = "registration"

_SCHEMA: str = """
CREATE TABLE IF NOT EXISTS review_pair (
    id INTEGER PRIMARY KEY,
    language TEXT NOT NULL,
    decade INTEGER,
    score REAL NOT NULL,
    band TEXT NOT NULL,
    source TEXT NOT NULL,
    pairing_type TEXT NOT NULL DEFAULT 'registration',
    marc_control_id TEXT NOT NULL,
    marc_json TEXT NOT NULL,
    marc_title TEXT,
    marc_author TEXT,
    marc_publisher TEXT,
    marc_year INTEGER,
    nypl_uuid TEXT NOT NULL,
    cce_title TEXT,
    cce_author TEXT,
    cce_publishers TEXT,
    cce_claimants TEXT,
    cce_reg_year INTEGER,
    cce_was_renewed INTEGER,
    cce_regnum TEXT,
    cce_edition TEXT,
    cce_publication_places TEXT,
    cce_author_place TEXT,
    cce_author_is_claimant INTEGER,
    cce_copies TEXT,
    cce_aff_date TEXT,
    cce_desc TEXT,
    cce_notes TEXT,
    cce_new_matter_claimed TEXT,
    cce_copy_date TEXT,
    cce_notice_date TEXT,
    cce_lccn TEXT,
    cce_prev_regnums TEXT,
    cce_renewal_id TEXT,
    cce_renewal_oreg TEXT,
    cce_renewal_rdat TEXT,
    cce_renewal_author TEXT,
    cce_renewal_title TEXT,
    cce_renewal_claimants TEXT,
    cce_renewal_new_matter TEXT,
    evidence_json TEXT NOT NULL,
    evidence_sources_json TEXT NOT NULL DEFAULT '{}',
    audit_note TEXT,
    alt_pair_id INTEGER,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS label (
    id INTEGER PRIMARY KEY,
    pair_id INTEGER NOT NULL REFERENCES review_pair(id),
    verdict TEXT NOT NULL CHECK (verdict IN ('match', 'no_match', 'unsure')),
    note TEXT,
    labeled_at TEXT NOT NULL,
    categories TEXT NOT NULL DEFAULT '[]'
);

CREATE INDEX IF NOT EXISTS ix_review_pair_lang_band ON review_pair (language, band);
CREATE INDEX IF NOT EXISTS ix_label_pair ON label (pair_id);
"""


class LabelInsertResult(Struct, frozen=True, forbid_unknown_fields=True):
    """Identifier and timestamp for one freshly inserted ``label`` row."""

    label_id: int
    labeled_at: str


class CurrentLabelRow(Struct, frozen=True, forbid_unknown_fields=True):
    """The current verdict for one pair joined with its MARC blob.

    Also carries the CCE-side identifiers (``cce_regnum``, ``cce_renewal_id``,
    ``cce_renewal_oreg``) so callers like ``seed-vault`` can persist the full
    schema-4 :class:`~pd_groundtruth.label_vault.VaultEntry` without a second
    lookup against the CCE index.
    """

    pair_id: int
    marc_control_id: str
    nypl_uuid: str
    marc_json: str
    verdict: str
    note: str | None
    labeled_at: str
    cce_regnum: str | None = None
    cce_renewal_id: str | None = None
    cce_renewal_oreg: str | None = None


class LabeledPairRow(Struct, frozen=True, forbid_unknown_fields=True):
    """A flat projection of one current-label row for the ``/labels`` table.

    Carries only the columns the table view renders (no MARC blob, no
    evidence): the originating pair id and its denormalized ``marc_*`` /
    ``cce_title`` columns, the current verdict and its optional note, the
    structured ``categories`` for chip rendering, plus the ``labeled_at``
    timestamp for relative-time rendering.
    """

    pair_id: int
    language: str
    marc_control_id: str
    marc_title: str | None
    cce_title: str | None
    verdict: str
    note: str | None
    labeled_at: str
    categories: tuple[CategoryKey, ...] = ()


SORT_DESC: str = "desc"
SORT_ASC: str = "asc"

_VALID_SORTS: frozenset[str] = frozenset({SORT_DESC, SORT_ASC})


class LabelFilters(Struct, frozen=True, forbid_unknown_fields=True):
    """Narrowing filters and ordering for :meth:`ReviewDb.iter_labeled_pairs`.

    The narrowing fields (``verdict``, ``language``, ``q``, ``categories``)
    AND together; an unset field imposes no constraint. ``q`` is a
    case-insensitive substring matched against ``marc_title``, ``cce_title``,
    and ``marc_control_id``. ``categories`` is a tuple of vocabulary keys; a
    row matches when the stored ``label.categories`` JSON array contains *any*
    of the selected keys (OR semantics within the category dimension).
    ``sort`` selects the ``labeled_at`` ordering: ``"desc"`` (default) puts
    the most recently labeled pair first, ``"asc"`` puts the oldest first.
    """

    verdict: str | None = None
    language: str | None = None
    q: str | None = None
    sort: str = SORT_DESC
    categories: tuple[CategoryKey, ...] = ()


_NO_LABEL_FILTERS: LabelFilters = LabelFilters()


class PairInsert(Struct, frozen=True, forbid_unknown_fields=True):
    """All column values for one :func:`ReviewDb.insert_pair` call."""

    language: str
    decade: int | None
    score: float
    band: str
    source: str
    marc_control_id: str
    marc_json: str
    marc_title: str | None
    marc_author: str | None
    marc_publisher: str | None
    marc_year: int | None
    nypl_uuid: str
    cce_title: str | None
    cce_author: str | None
    cce_publishers: str | None
    cce_claimants: str | None
    cce_reg_year: int | None
    cce_was_renewed: bool | None
    cce_regnum: str | None
    evidence_json: str
    cce_edition: str | None = None
    cce_publication_places: str | None = None
    cce_author_place: str | None = None
    cce_author_is_claimant: bool | None = None
    cce_copies: str | None = None
    cce_aff_date: str | None = None
    cce_desc: str | None = None
    cce_notes: str | None = None
    cce_new_matter_claimed: str | None = None
    cce_copy_date: str | None = None
    cce_notice_date: str | None = None
    cce_lccn: str | None = None
    cce_prev_regnums: str | None = None
    cce_renewal_id: str | None = None
    cce_renewal_oreg: str | None = None
    cce_renewal_rdat: str | None = None
    cce_renewal_author: str | None = None
    cce_renewal_title: str | None = None
    cce_renewal_claimants: str | None = None
    cce_renewal_new_matter: str | None = None
    evidence_sources_json: str = "{}"
    audit_note: str | None = None
    pairing_type: str = PAIRING_REGISTRATION
    alt_pair_id: int | None = None


class ReviewPairRow(Struct, frozen=True, forbid_unknown_fields=True):
    """One persisted ``review_pair`` row, fully typed for downstream use."""

    id: int
    language: str
    decade: int | None
    score: float
    band: str
    source: str
    marc_control_id: str
    marc_json: str
    marc_title: str | None
    marc_author: str | None
    marc_publisher: str | None
    marc_year: int | None
    nypl_uuid: str
    cce_title: str | None
    cce_author: str | None
    cce_publishers: str | None
    cce_claimants: str | None
    cce_reg_year: int | None
    cce_was_renewed: int | None
    cce_regnum: str | None
    evidence_json: str
    created_at: str
    cce_edition: str | None = None
    cce_publication_places: str | None = None
    cce_author_place: str | None = None
    cce_author_is_claimant: int | None = None
    cce_copies: str | None = None
    cce_aff_date: str | None = None
    cce_desc: str | None = None
    cce_notes: str | None = None
    cce_new_matter_claimed: str | None = None
    cce_copy_date: str | None = None
    cce_notice_date: str | None = None
    cce_lccn: str | None = None
    cce_prev_regnums: str | None = None
    cce_renewal_id: str | None = None
    cce_renewal_oreg: str | None = None
    cce_renewal_rdat: str | None = None
    cce_renewal_author: str | None = None
    cce_renewal_title: str | None = None
    cce_renewal_claimants: str | None = None
    cce_renewal_new_matter: str | None = None
    evidence_sources_json: str = "{}"
    audit_note: str | None = None
    pairing_type: str = PAIRING_REGISTRATION
    alt_pair_id: int | None = None


class LanguageProgress(Struct, frozen=True, forbid_unknown_fields=True):
    """Per-language pair totals and labeled counts for the stats page."""

    language: str
    total: int
    labeled: int


class ProgressCounts(Struct, frozen=True, forbid_unknown_fields=True):
    """A running tally of review progress across the whole queue.

    ``labeled`` counts distinct ``pair_id`` values with at least one ``label``
    row; ``match`` / ``no_match`` / ``unsure`` count pairs by their *current*
    verdict (the latest label by ``MAX(id)``, the monotonic action order), so
    re-labels move a pair between buckets without double-counting.
    """

    total: int
    labeled: int
    remaining: int
    match: int
    no_match: int
    unsure: int
    by_language: tuple[LanguageProgress, ...]


def _now() -> str:
    """Return the current UTC instant as an ISO-8601 string."""
    return datetime.now(UTC).isoformat()


def _encode_categories(categories: tuple[CategoryKey, ...]) -> str:
    """Serialize a categories tuple to the JSON TEXT stored in ``label.categories``."""
    return json_encode(categories).decode("utf-8")


def _decode_categories(raw: str | None) -> tuple[CategoryKey, ...]:
    """Decode a stored ``label.categories`` TEXT back into a typed tuple.

    Returns an empty tuple for the ``'[]'`` default; raises
    :class:`msgspec.ValidationError` if the persisted JSON carries an
    unknown :data:`CategoryKey` value so corruption surfaces at read time
    rather than propagating downstream.
    """
    if not raw:
        return ()
    return json_decode(raw.encode("utf-8"), type=tuple[CategoryKey, ...])


_ADDITIVE_CCE_COLUMNS: tuple[tuple[str, str], ...] = (
    ("cce_edition", "TEXT"),
    ("cce_publication_places", "TEXT"),
    ("cce_author_place", "TEXT"),
    ("cce_author_is_claimant", "INTEGER"),
    ("cce_copies", "TEXT"),
    ("cce_aff_date", "TEXT"),
    ("cce_desc", "TEXT"),
    ("cce_notes", "TEXT"),
    ("cce_new_matter_claimed", "TEXT"),
    ("cce_copy_date", "TEXT"),
    ("cce_notice_date", "TEXT"),
    ("cce_lccn", "TEXT"),
    ("cce_prev_regnums", "TEXT"),
    ("cce_renewal_id", "TEXT"),
    ("cce_renewal_oreg", "TEXT"),
    ("cce_renewal_rdat", "TEXT"),
    ("cce_renewal_author", "TEXT"),
    ("cce_renewal_title", "TEXT"),
    ("cce_renewal_claimants", "TEXT"),
    ("cce_renewal_new_matter", "TEXT"),
    ("evidence_sources_json", "TEXT NOT NULL DEFAULT '{}'"),
    ("audit_note", "TEXT"),
    ("pairing_type", "TEXT NOT NULL DEFAULT 'registration'"),
    ("alt_pair_id", "INTEGER"),
)


def _row_to_pair(row: Row) -> ReviewPairRow:
    """Map a ``review_pair`` :class:`sqlite3.Row` to a typed struct."""
    return ReviewPairRow(
        id=row["id"],
        language=row["language"],
        decade=row["decade"],
        score=row["score"],
        band=row["band"],
        source=row["source"],
        marc_control_id=row["marc_control_id"],
        marc_json=row["marc_json"],
        marc_title=row["marc_title"],
        marc_author=row["marc_author"],
        marc_publisher=row["marc_publisher"],
        marc_year=row["marc_year"],
        nypl_uuid=row["nypl_uuid"],
        cce_title=row["cce_title"],
        cce_author=row["cce_author"],
        cce_publishers=row["cce_publishers"],
        cce_claimants=row["cce_claimants"],
        cce_reg_year=row["cce_reg_year"],
        cce_was_renewed=row["cce_was_renewed"],
        cce_regnum=row["cce_regnum"],
        evidence_json=row["evidence_json"],
        created_at=row["created_at"],
        cce_edition=row["cce_edition"],
        cce_publication_places=row["cce_publication_places"],
        cce_author_place=row["cce_author_place"],
        cce_author_is_claimant=row["cce_author_is_claimant"],
        cce_copies=row["cce_copies"],
        cce_aff_date=row["cce_aff_date"],
        cce_desc=row["cce_desc"],
        cce_notes=row["cce_notes"],
        cce_new_matter_claimed=row["cce_new_matter_claimed"],
        cce_copy_date=row["cce_copy_date"],
        cce_notice_date=row["cce_notice_date"],
        cce_lccn=row["cce_lccn"],
        cce_prev_regnums=row["cce_prev_regnums"],
        cce_renewal_id=row["cce_renewal_id"],
        cce_renewal_oreg=row["cce_renewal_oreg"],
        cce_renewal_rdat=row["cce_renewal_rdat"],
        cce_renewal_author=row["cce_renewal_author"],
        cce_renewal_title=row["cce_renewal_title"],
        cce_renewal_claimants=row["cce_renewal_claimants"],
        cce_renewal_new_matter=row["cce_renewal_new_matter"],
        evidence_sources_json=row["evidence_sources_json"] or "{}",
        audit_note=row["audit_note"],
        pairing_type=row["pairing_type"],
        alt_pair_id=row["alt_pair_id"],
    )


class ReviewDb:
    """Typed connection to the SQLite review database.

    Use as a context manager so the underlying connection is committed and
    closed deterministically. All query methods return frozen msgspec
    structs (or ``None``); no method exposes ``Any``.
    """

    __slots__ = ("_conn",)

    def __init__(self, connection: Connection) -> None:
        self._conn = connection
        self._conn.row_factory = Row

    @classmethod
    def connect(cls, path: Path) -> ReviewDb:
        """Open (creating if absent) the database at ``path`` and init schema."""
        connection = sqlite_connect(path)
        db = cls(connection)
        db.init_schema()
        return db

    def __enter__(self) -> ReviewDb:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: object | None,
    ) -> None:
        if exc_type is None:
            self._conn.commit()
        self._conn.close()

    def init_schema(self) -> None:
        """Create tables and indices if they do not already exist.

        Also adds the extended ``cce_*`` columns to ``review_pair`` on databases
        created before the full CCE surface was carried into the queue. Existing
        rows get ``NULL`` for the new columns; the card view guards every new
        field with an "if set" check.

        Legacy ``label_reason`` and ``label_field_annotation`` tables from
        prior schema versions are left in place when present; SQLite makes
        column/table drops painful and the new code never queries them. Use
        ``build-queue --rebuild`` to get a clean schema next time.
        """
        self._conn.executescript(_SCHEMA)
        pair_columns = {row["name"] for row in self._conn.execute("PRAGMA table_info(review_pair)")}
        for column_name, column_type in _ADDITIVE_CCE_COLUMNS:
            if column_name not in pair_columns:
                self._conn.execute(
                    f"ALTER TABLE review_pair ADD COLUMN {column_name} {column_type}"
                )

    def commit(self) -> None:
        """Flush pending writes to disk."""
        self._conn.commit()

    def insert_pair(self, pair: PairInsert) -> int:
        """Insert one proposed pair and return its new ``review_pair.id``."""
        cursor = self._conn.execute(
            """
            INSERT INTO review_pair (
                language, decade, score, band, source, pairing_type, marc_control_id,
                marc_json, marc_title, marc_author, marc_publisher, marc_year,
                nypl_uuid, cce_title, cce_author, cce_publishers, cce_claimants,
                cce_reg_year, cce_was_renewed, cce_regnum, evidence_json,
                cce_edition, cce_publication_places, cce_author_place,
                cce_author_is_claimant, cce_copies, cce_aff_date, cce_desc,
                cce_notes, cce_new_matter_claimed, cce_copy_date, cce_notice_date,
                cce_lccn, cce_prev_regnums,
                cce_renewal_id, cce_renewal_oreg, cce_renewal_rdat,
                cce_renewal_author, cce_renewal_title, cce_renewal_claimants,
                cce_renewal_new_matter,
                evidence_sources_json,
                audit_note,
                alt_pair_id,
                created_at
            ) VALUES (
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                ?, ?,
                ?, ?, ?,
                ?, ?, ?,
                ?,
                ?,
                ?,
                ?,
                ?
            )
            """,
            (
                pair.language,
                pair.decade,
                pair.score,
                pair.band,
                pair.source,
                pair.pairing_type,
                pair.marc_control_id,
                pair.marc_json,
                pair.marc_title,
                pair.marc_author,
                pair.marc_publisher,
                pair.marc_year,
                pair.nypl_uuid,
                pair.cce_title,
                pair.cce_author,
                pair.cce_publishers,
                pair.cce_claimants,
                pair.cce_reg_year,
                None if pair.cce_was_renewed is None else int(pair.cce_was_renewed),
                pair.cce_regnum,
                pair.evidence_json,
                pair.cce_edition,
                pair.cce_publication_places,
                pair.cce_author_place,
                None if pair.cce_author_is_claimant is None else int(pair.cce_author_is_claimant),
                pair.cce_copies,
                pair.cce_aff_date,
                pair.cce_desc,
                pair.cce_notes,
                pair.cce_new_matter_claimed,
                pair.cce_copy_date,
                pair.cce_notice_date,
                pair.cce_lccn,
                pair.cce_prev_regnums,
                pair.cce_renewal_id,
                pair.cce_renewal_oreg,
                pair.cce_renewal_rdat,
                pair.cce_renewal_author,
                pair.cce_renewal_title,
                pair.cce_renewal_claimants,
                pair.cce_renewal_new_matter,
                pair.evidence_sources_json,
                pair.audit_note,
                pair.alt_pair_id,
                _now(),
            ),
        )
        row_id = cursor.lastrowid
        if row_id is None:  # pragma: no cover
            raise RuntimeError("INSERT did not return a rowid")
        return row_id

    def stratum_counts(self) -> dict[tuple[str, str], int]:
        """Return persisted pair counts keyed on ``(language, band)``."""
        rows = self._conn.execute(
            "SELECT language, band, COUNT(*) AS n FROM review_pair GROUP BY language, band"
        ).fetchall()
        return {(row["language"], row["band"]): row["n"] for row in rows}

    def pair_keys(self) -> set[tuple[str, str]]:
        """Return every ``(marc_control_id, nypl_uuid)`` already in ``review_pair``.

        Used by the ``vault-into-queue`` backfill to compute the set of vault
        entries that need to be inserted into the rebuilt queue: anything not in
        this set is missing and needs a row.
        """
        rows = self._conn.execute("SELECT marc_control_id, nypl_uuid FROM review_pair").fetchall()
        return {(row["marc_control_id"], row["nypl_uuid"]) for row in rows}

    def next_unlabeled(
        self,
        *,
        language: str | None = None,
        band: str | None = None,
        exclude_pair_ids: tuple[int, ...] = (),
    ) -> ReviewPairRow | None:
        """Return the lowest-id ``review_pair`` with no current label.

        A pair is considered labeled once any ``label`` row references it
        (Phase 2b appends re-labels rather than deleting), so "unlabeled"
        means *no* label rows exist. Optional ``language`` / ``band``
        filters narrow the queue for focused review sessions.
        ``exclude_pair_ids`` lets a caller skip an explicit set of ids (used
        by the review UI to remember session-local skips so the Skip button
        advances past the current pair without persisting that decision).
        """
        clauses: list[str] = [
            "rp.id NOT IN (SELECT pair_id FROM label)",
        ]
        params: list[str | int] = []
        if language is not None:
            clauses.append("rp.language = ?")
            params.append(language)
        if band is not None:
            clauses.append("rp.band = ?")
            params.append(band)
        if exclude_pair_ids:
            placeholders = ",".join("?" * len(exclude_pair_ids))
            clauses.append(f"rp.id NOT IN ({placeholders})")
            params.extend(exclude_pair_ids)
        where = " AND ".join(clauses)
        row = self._conn.execute(
            f"SELECT rp.* FROM review_pair rp WHERE {where} ORDER BY rp.id LIMIT 1",
            params,
        ).fetchone()
        if row is None:
            return None
        return _row_to_pair(row)

    def get_pair(self, pair_id: int) -> ReviewPairRow | None:
        """Return one ``review_pair`` by id, or ``None`` if it does not exist."""
        row = self._conn.execute(
            "SELECT * FROM review_pair WHERE id = ?",
            (pair_id,),
        ).fetchone()
        if row is None:
            return None
        return _row_to_pair(row)

    def get_current_label(self, pair_id: int) -> CurrentLabelRow | None:
        """Return the latest verdict + note for ``pair_id``, or ``None`` if unlabeled.

        Single-row variant of :meth:`iter_current_labels` used by the review
        card to pre-fill the note textarea and visually mark the current
        verdict button when revisiting an already-labeled pair. The "current"
        label is the one with ``MAX(id)`` per ``pair_id`` (the monotonic action
        order used elsewhere), so re-labels are reflected immediately.
        """
        row = self._conn.execute(
            """
            SELECT
                rp.id AS pair_id,
                rp.marc_control_id AS marc_control_id,
                rp.nypl_uuid AS nypl_uuid,
                rp.marc_json AS marc_json,
                rp.cce_regnum AS cce_regnum,
                rp.cce_renewal_id AS cce_renewal_id,
                rp.cce_renewal_oreg AS cce_renewal_oreg,
                cur.verdict AS verdict,
                cur.note AS note,
                cur.labeled_at AS labeled_at
            FROM review_pair rp
            JOIN (
                SELECT l.id, l.pair_id, l.verdict, l.note, l.labeled_at
                FROM label l
                JOIN (
                    SELECT pair_id, MAX(id) AS max_id FROM label WHERE pair_id = ? GROUP BY pair_id
                ) latest
                  ON l.pair_id = latest.pair_id AND l.id = latest.max_id
            ) cur
              ON cur.pair_id = rp.id
            WHERE rp.id = ?
            """,
            (pair_id, pair_id),
        ).fetchone()
        if row is None:
            return None
        return CurrentLabelRow(
            pair_id=row["pair_id"],
            marc_control_id=row["marc_control_id"],
            nypl_uuid=row["nypl_uuid"],
            marc_json=row["marc_json"],
            verdict=row["verdict"],
            note=row["note"],
            labeled_at=row["labeled_at"],
            cce_regnum=row["cce_regnum"],
            cce_renewal_id=row["cce_renewal_id"],
            cce_renewal_oreg=row["cce_renewal_oreg"],
        )

    def progress(self) -> ProgressCounts:
        """Return a running tally of labeling progress across the queue.

        Counts distinct labeled pairs and groups current verdicts (the latest
        label per ``pair_id``) into match / no_match / unsure buckets, plus a
        per-language total/labeled breakdown for the stats page.
        """
        total = self._conn.execute("SELECT COUNT(*) AS n FROM review_pair").fetchone()["n"]
        labeled = self._conn.execute("SELECT COUNT(DISTINCT pair_id) AS n FROM label").fetchone()[
            "n"
        ]
        verdict_rows = self._conn.execute(
            """
            SELECT cur.verdict AS verdict, COUNT(*) AS n
            FROM (
                SELECT l.pair_id, l.verdict
                FROM label l
                JOIN (
                    SELECT pair_id, MAX(id) AS max_id
                    FROM label GROUP BY pair_id
                ) latest
                  ON l.pair_id = latest.pair_id AND l.id = latest.max_id
            ) cur
            GROUP BY cur.verdict
            """
        ).fetchall()
        by_verdict = {row["verdict"]: row["n"] for row in verdict_rows}
        language_rows = self._conn.execute(
            """
            SELECT rp.language AS language,
                   COUNT(*) AS total,
                   COUNT(DISTINCT l.pair_id) AS labeled
            FROM review_pair rp
            LEFT JOIN label l ON l.pair_id = rp.id
            GROUP BY rp.language
            ORDER BY rp.language
            """
        ).fetchall()
        by_language = tuple(
            LanguageProgress(
                language=row["language"],
                total=row["total"],
                labeled=row["labeled"],
            )
            for row in language_rows
        )
        return ProgressCounts(
            total=total,
            labeled=labeled,
            remaining=total - labeled,
            match=by_verdict.get(VERDICT_MATCH, 0),
            no_match=by_verdict.get(VERDICT_NO_MATCH, 0),
            unsure=by_verdict.get(VERDICT_UNSURE, 0),
            by_language=by_language,
        )

    def add_label(
        self,
        pair_id: int,
        verdict: str,
        note: str | None = None,
        categories: tuple[CategoryKey, ...] = (),
    ) -> LabelInsertResult:
        """Append a verdict for ``pair_id`` and return the new label's id + timestamp.

        ``note`` is optional free text — the labeler's open rationale.
        ``categories`` (default ``()``) is the structured complement: a tuple
        of vocabulary keys recording recurring rationale patterns; it is
        encoded as a JSON array TEXT alongside the verdict so the DB stays
        symmetric with the label vault. The returned
        :class:`LabelInsertResult` exposes the ISO-8601 timestamp stamped onto
        the row so the caller (review UI) can pass the exact same value to the
        label vault and keep DB and vault in lockstep.

        Raises:
            ValueError: If ``verdict`` is not one of ``match``,
                ``no_match``, or ``unsure``.
        """
        if verdict not in _VALID_VERDICTS:
            raise ValueError(f"invalid verdict {verdict!r}")
        labeled_at = _now()
        cursor = self._conn.execute(
            "INSERT INTO label (pair_id, verdict, note, labeled_at, categories)"
            " VALUES (?, ?, ?, ?, ?)",
            (pair_id, verdict, note, labeled_at, _encode_categories(categories)),
        )
        row_id = cursor.lastrowid
        if row_id is None:  # pragma: no cover
            raise RuntimeError("INSERT did not return a rowid")
        return LabelInsertResult(label_id=row_id, labeled_at=labeled_at)

    def iter_current_labels(self) -> Iterator[CurrentLabelRow]:
        """Yield one :class:`CurrentLabelRow` per labeled pair, latest verdict only.

        The "current" label is the one with ``MAX(id)`` per ``pair_id`` (the
        monotonic action order used elsewhere).
        """
        rows = self._conn.execute(
            """
            SELECT
                rp.id AS pair_id,
                rp.marc_control_id AS marc_control_id,
                rp.nypl_uuid AS nypl_uuid,
                rp.marc_json AS marc_json,
                rp.cce_regnum AS cce_regnum,
                rp.cce_renewal_id AS cce_renewal_id,
                rp.cce_renewal_oreg AS cce_renewal_oreg,
                cur.verdict AS verdict,
                cur.note AS note,
                cur.labeled_at AS labeled_at
            FROM review_pair rp
            JOIN (
                SELECT l.id, l.pair_id, l.verdict, l.note, l.labeled_at
                FROM label l
                JOIN (
                    SELECT pair_id, MAX(id) AS max_id FROM label GROUP BY pair_id
                ) latest
                  ON l.pair_id = latest.pair_id AND l.id = latest.max_id
            ) cur
              ON cur.pair_id = rp.id
            ORDER BY rp.id
            """
        ).fetchall()
        for row in rows:
            yield CurrentLabelRow(
                pair_id=row["pair_id"],
                marc_control_id=row["marc_control_id"],
                nypl_uuid=row["nypl_uuid"],
                marc_json=row["marc_json"],
                verdict=row["verdict"],
                note=row["note"],
                labeled_at=row["labeled_at"],
                cce_regnum=row["cce_regnum"],
                cce_renewal_id=row["cce_renewal_id"],
                cce_renewal_oreg=row["cce_renewal_oreg"],
            )

    def insert_existing_label(
        self,
        pair_id: int,
        verdict: str,
        labeled_at: str,
        note: str | None = None,
        categories: tuple[CategoryKey, ...] = (),
    ) -> int:
        """Insert a label whose verdict came from outside this database.

        Used by ``build-queue`` to pre-apply labels carried over from the label
        vault: the verdict / note / ``labeled_at`` / ``categories`` come from
        the vault entry verbatim, so a rebuilt queue still reports the pair as
        labeled without inventing a new timestamp and the structured
        rationale tags survive the rebuild.

        Raises:
            ValueError: If ``verdict`` is not one of ``match``,
                ``no_match``, or ``unsure``.
        """
        if verdict not in _VALID_VERDICTS:
            raise ValueError(f"invalid verdict {verdict!r}")
        cursor = self._conn.execute(
            "INSERT INTO label (pair_id, verdict, note, labeled_at, categories)"
            " VALUES (?, ?, ?, ?, ?)",
            (pair_id, verdict, note, labeled_at, _encode_categories(categories)),
        )
        row_id = cursor.lastrowid
        if row_id is None:  # pragma: no cover
            raise RuntimeError("INSERT did not return a rowid")
        return row_id

    def count_labeled_pairs(self, filters: LabelFilters = _NO_LABEL_FILTERS) -> int:
        """Return the number of current-label rows matching ``filters``.

        Counts distinct ``pair_id`` values whose latest label satisfies every
        active filter (matches the row set :meth:`iter_labeled_pairs` would
        yield without pagination). Used by the ``/labels`` table view to size
        the pager and the "showing N of M" header.
        """
        sql, params = self._labeled_pairs_query(
            filters=filters,
            select="COUNT(*) AS n",
            order_limit="",
        )
        row = self._conn.execute(sql, params).fetchone()
        return int(row["n"])

    def iter_labeled_pairs(
        self,
        filters: LabelFilters = _NO_LABEL_FILTERS,
        *,
        page_size: int = 100,
        page: int = 1,
    ) -> tuple[LabeledPairRow, ...]:
        """Return one :class:`LabeledPairRow` per current label, paged and filtered.

        The "current" label is the latest by ``MAX(id)`` per ``pair_id``
        (matching :meth:`iter_current_labels` / :meth:`progress`). Rows are
        ordered by ``labeled_at`` in the direction selected by
        ``filters.sort`` — ``"desc"`` (default) puts the most recently labeled
        pair first, ``"asc"`` puts the oldest first — then sliced by
        ``page_size`` / ``page`` (1-indexed). The ``rp.id`` tie-breaker
        mirrors the primary direction so equal timestamps stay in a stable,
        ordering-consistent order across both sorts.
        """
        if page < 1:
            raise ValueError(f"page must be >= 1, got {page}")
        if page_size < 1:
            raise ValueError(f"page_size must be >= 1, got {page_size}")
        if filters.sort not in _VALID_SORTS:
            raise ValueError(f"invalid sort {filters.sort!r}")
        offset = (page - 1) * page_size
        direction = "DESC" if filters.sort == SORT_DESC else "ASC"
        sql, params = self._labeled_pairs_query(
            filters=filters,
            select=(
                "rp.id AS pair_id, rp.language AS language, "
                "rp.marc_control_id AS marc_control_id, rp.marc_title AS marc_title, "
                "rp.cce_title AS cce_title, l.verdict AS verdict, "
                "l.note AS note, l.labeled_at AS labeled_at, "
                "l.categories AS categories"
            ),
            order_limit=f"ORDER BY l.labeled_at {direction}, rp.id {direction} LIMIT ? OFFSET ?",
        )
        params = (*params, page_size, offset)
        rows = self._conn.execute(sql, params).fetchall()
        return tuple(
            LabeledPairRow(
                pair_id=row["pair_id"],
                language=row["language"],
                marc_control_id=row["marc_control_id"],
                marc_title=row["marc_title"],
                cce_title=row["cce_title"],
                verdict=row["verdict"],
                note=row["note"],
                labeled_at=row["labeled_at"],
                categories=_decode_categories(row["categories"]),
            )
            for row in rows
        )

    def _labeled_pairs_query(
        self,
        *,
        filters: LabelFilters,
        select: str,
        order_limit: str,
    ) -> tuple[str, tuple[str | int, ...]]:
        """Assemble the WHERE clause and params shared by count and list queries."""
        clauses: list[str] = []
        params: list[str | int] = []
        if filters.verdict is not None:
            clauses.append("l.verdict = ?")
            params.append(filters.verdict)
        if filters.language is not None:
            clauses.append("rp.language = ?")
            params.append(filters.language)
        if filters.q:
            pattern = f"%{filters.q.lower()}%"
            clauses.append(
                "(lower(COALESCE(rp.marc_title, '')) LIKE ?"
                " OR lower(COALESCE(rp.cce_title, '')) LIKE ?"
                " OR lower(rp.marc_control_id) LIKE ?)"
            )
            params.extend((pattern, pattern, pattern))
        if filters.categories:
            placeholders = ",".join("?" * len(filters.categories))
            clauses.append(
                "EXISTS ("
                "SELECT 1 FROM json_each(l.categories) "
                f"WHERE json_each.value IN ({placeholders})"
                ")"
            )
            params.extend(filters.categories)
        where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
        sql = f"""
            SELECT {select}
            FROM (
                SELECT pair_id, MAX(id) AS last_id FROM label GROUP BY pair_id
            ) cl
            JOIN label l ON l.id = cl.last_id
            JOIN review_pair rp ON rp.id = cl.pair_id
            {where}
            {order_limit}
        """
        return sql, tuple(params)


__all__ = [
    "PAIRING_REGISTRATION",
    "SORT_ASC",
    "SORT_DESC",
    "VERDICT_MATCH",
    "VERDICT_NO_MATCH",
    "VERDICT_UNSURE",
    "CurrentLabelRow",
    "LabelFilters",
    "LabelInsertResult",
    "LabeledPairRow",
    "LanguageProgress",
    "PairInsert",
    "ProgressCounts",
    "ReviewDb",
    "ReviewPairRow",
]
