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

from datetime import UTC
from datetime import datetime
from pathlib import Path
from sqlite3 import Connection
from sqlite3 import Row
from sqlite3 import connect as sqlite_connect

from msgspec import Struct

VERDICT_MATCH: str = "match"
VERDICT_NO_MATCH: str = "no_match"
VERDICT_UNSURE: str = "unsure"

_VALID_VERDICTS: frozenset[str] = frozenset({VERDICT_MATCH, VERDICT_NO_MATCH, VERDICT_UNSURE})

_SCHEMA: str = """
CREATE TABLE IF NOT EXISTS review_pair (
    id INTEGER PRIMARY KEY,
    language TEXT NOT NULL,
    decade INTEGER,
    score REAL NOT NULL,
    band TEXT NOT NULL,
    source TEXT NOT NULL,
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
    evidence_json TEXT NOT NULL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS label (
    id INTEGER PRIMARY KEY,
    pair_id INTEGER NOT NULL REFERENCES review_pair(id),
    verdict TEXT NOT NULL CHECK (verdict IN ('match', 'no_match', 'unsure')),
    note TEXT,
    labeled_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS ix_review_pair_lang_band ON review_pair (language, band);
CREATE INDEX IF NOT EXISTS ix_label_pair ON label (pair_id);
"""


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


class LanguageProgress(Struct, frozen=True, forbid_unknown_fields=True):
    """Per-language pair totals and labeled counts for the stats page."""

    language: str
    total: int
    labeled: int


class ProgressCounts(Struct, frozen=True, forbid_unknown_fields=True):
    """A running tally of review progress across the whole queue.

    ``labeled`` counts distinct ``pair_id`` values with at least one ``label``
    row; ``match`` / ``no_match`` / ``unsure`` count pairs by their *current*
    verdict (the latest label by ``(labeled_at, id)``), so re-labels move a
    pair between buckets without double-counting.
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
        """Create tables and indices if they do not already exist."""
        self._conn.executescript(_SCHEMA)

    def commit(self) -> None:
        """Flush pending writes to disk."""
        self._conn.commit()

    def insert_pair(self, pair: PairInsert) -> int:
        """Insert one proposed pair and return its new ``review_pair.id``."""
        cursor = self._conn.execute(
            """
            INSERT INTO review_pair (
                language, decade, score, band, source, marc_control_id,
                marc_json, marc_title, marc_author, marc_publisher, marc_year,
                nypl_uuid, cce_title, cce_author, cce_publishers, cce_claimants,
                cce_reg_year, cce_was_renewed, cce_regnum, evidence_json, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                pair.language,
                pair.decade,
                pair.score,
                pair.band,
                pair.source,
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

    def next_unlabeled(
        self,
        *,
        language: str | None = None,
        band: str | None = None,
    ) -> ReviewPairRow | None:
        """Return the lowest-id ``review_pair`` with no current label.

        A pair is considered labeled once any ``label`` row references it
        (Phase 2b appends re-labels rather than deleting), so "unlabeled"
        means *no* label rows exist. Optional ``language`` / ``band``
        filters narrow the queue for focused review sessions.
        """
        clauses: list[str] = [
            "rp.id NOT IN (SELECT pair_id FROM label)",
        ]
        params: list[str] = []
        if language is not None:
            clauses.append("rp.language = ?")
            params.append(language)
        if band is not None:
            clauses.append("rp.band = ?")
            params.append(band)
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
                    SELECT pair_id, MAX(labeled_at) AS max_at, MAX(id) AS max_id
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

    def add_label(self, pair_id: int, verdict: str, note: str | None = None) -> int:
        """Append a verdict for ``pair_id`` and return the new ``label.id``.

        Raises:
            ValueError: If ``verdict`` is not one of ``match``,
                ``no_match``, or ``unsure``.
        """
        if verdict not in _VALID_VERDICTS:
            raise ValueError(f"invalid verdict {verdict!r}")
        cursor = self._conn.execute(
            "INSERT INTO label (pair_id, verdict, note, labeled_at) VALUES (?, ?, ?, ?)",
            (pair_id, verdict, note, _now()),
        )
        row_id = cursor.lastrowid
        if row_id is None:  # pragma: no cover
            raise RuntimeError("INSERT did not return a rowid")
        return row_id


__all__ = [
    "VERDICT_MATCH",
    "VERDICT_NO_MATCH",
    "VERDICT_UNSURE",
    "LanguageProgress",
    "PairInsert",
    "ProgressCounts",
    "ReviewDb",
    "ReviewPairRow",
]
