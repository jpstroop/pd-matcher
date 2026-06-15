"""Phase-0 characterization of the volume.compat scorer on whole/part cases (#82).

Throwaway investigate-first measurement script. NOT shipped; ``scripts/`` is
gitignored from the published package (a ``!`` exception lets the maintainer
commit this proof). It does NOT modify anything under ``src/`` and writes
nothing under ``data/``; the vault is read-only.

WHAT THIS RESOLVES. Issue #82 hypothesizes the existing ``volume.compat``
scorer fails on whole/part disagreements for three nameable reasons:
(1) the signal lives in fields the scorer never reads (MARC 500 notes,
edition, series; CCE notes, new_matter); (2) Roman vs Arabic part numbers
compare unequal (``"ii" != "2"``); (3) the scorer is language-blind. Before
spending Phase-1 effort, this script runs the CURRENT scorer AS-IS over the
whole/part vault slice and attributes every non-correct case to a CAUSE and a
FIELD, producing a ranked, data-justified Phase-1 priority list.

WHAT THIS SCRIPT DOES (READ-ONLY against vault + pool + LMDB index).

1. **Select the study slice.** From ``current_entries(vault)``: entries tagged
   ``marc_whole_cce_part`` or ``cce_whole_marc_part``, UNION a keyword pass over
   each entry's free-text ``note`` and its resolved record surface fields for
   whole/part designators (``v.``, ``vol``, ``pt.``, ``part``, ``tome``,
   ``tomo``, ``bd.``, ``band``, ``teil``, ``t. N``, ``heft``, ``fasc``,
   ``livre``, ``libro``, ``complete``, ``collected``, ``selected``, ``ser.``,
   ``aufl``, ``course N``).

2. **Resolve MARC + CCE records** per entry via the proven heldout path
   (``build_marc_index`` over the pool, ``NyplIndexLookup.get_registration``).
   Unresolved entries are counted and skipped.

3. **Run the CURRENT scorer** (``score_volume`` with a real ``ScorerContext``
   built by the production ``_build_context``, so ``ctx.language`` is correct)
   AND call ``_classify_marc`` / ``_classify_cce`` directly to capture
   ``(cardinality, part_number)`` per side.

4. **Classify each case's OUTCOME** against the gold verdict (see Method in the
   report): ``correctly_scored`` / ``skipped_but_signal_exists`` /
   ``misclassified``.

5. **Attribute a CAUSE** to every non-correct case by re-running detection over
   the UNREAD fields and normalization variants: ``signal_in_unread_field``,
   ``roman_numeral_miss``, ``word_number_miss``, ``pattern_gap_candidate``,
   ``genuinely_ambiguous``.

Usage:
    pdm run python scripts/volume_signal_characterization.py \\
        > docs/findings/volume_signal_characterization_2026-06-13.md

    # Smoke run capping the slice (report is flagged):
    pdm run python scripts/volume_signal_characterization.py --limit 30 \\
        > /tmp/volchar_smoke.md
"""

from __future__ import annotations

from argparse import ArgumentParser
from collections import Counter
from dataclasses import dataclass
from dataclasses import field
from pathlib import Path
from re import IGNORECASE
from re import compile as re_compile
from sys import stderr
from typing import Final

from pd_groundtruth.label_vault import VaultEntry
from pd_groundtruth.label_vault import current_entries
from pd_groundtruth.vault_pair_resolver import build_marc_index
from pd_matcher.cli import _load_default_matching_config
from pd_matcher.index.lookup import NyplIndexLookup
from pd_matcher.match.idf import build_author_idf_table
from pd_matcher.match.idf import build_idf_table
from pd_matcher.match.idf import build_publisher_idf_table
from pd_matcher.match.pipeline import _build_context
from pd_matcher.match.scorers.volume import _classify_cce
from pd_matcher.match.scorers.volume import _classify_marc
from pd_matcher.match.scorers.volume import _detect_part
from pd_matcher.match.scorers.volume import _is_collected_title
from pd_matcher.match.scorers.volume import _is_multivolume_whole
from pd_matcher.match.scorers.volume import score_volume
from pd_matcher.models import IndexedNyplRegRecord
from pd_matcher.models import MarcRecord
from pd_matcher.normalize.numbers import roman_to_arabic
from pd_matcher.normalize.numbers import word_to_int

_VAULT_PATH: Final[Path] = Path("data/label_vault.jsonl")
_POOL_PATH: Final[Path] = Path("data/candidates")
_INDEX_PATH: Final[Path] = Path("caches/cce.lmdb")
_PROGRESS_LOG: Final[Path] = Path("/tmp/agent-progress.log")

_DEFAULT_LANGUAGE: Final[str] = "eng"

_CAT_MARC_WHOLE: Final[str] = "marc_whole_cce_part"
_CAT_CCE_WHOLE: Final[str] = "cce_whole_marc_part"
_WHOLE_PART_CATEGORIES: Final[frozenset[str]] = frozenset(
    {_CAT_MARC_WHOLE, _CAT_CCE_WHOLE}
)

_VERDICT_MATCH: Final[str] = "match"
_VERDICT_NO_MATCH: Final[str] = "no_match"

# Cardinality labels (mirrors volume._Cardinality, which is private).
_WHOLE: Final[str] = "whole"
_WHOLE_OPEN: Final[str] = "whole_open"
_PART: Final[str] = "part"
_UNKNOWN: Final[str] = "unknown"

# Outcome labels.
_OUTCOME_CORRECT: Final[str] = "correctly_scored"
_OUTCOME_SKIPPED_SIGNAL: Final[str] = "skipped_but_signal_exists"
_OUTCOME_MISCLASSIFIED: Final[str] = "misclassified"

# Cause labels.
_CAUSE_UNREAD_FIELD: Final[str] = "signal_in_unread_field"
_CAUSE_ROMAN: Final[str] = "roman_numeral_miss"
_CAUSE_WORD: Final[str] = "word_number_miss"
_CAUSE_PATTERN_GAP: Final[str] = "pattern_gap_candidate"
_CAUSE_AMBIGUOUS: Final[str] = "genuinely_ambiguous"

# Field-source labels for the per-field signal-in-unread-field table.
_FIELD_MARC_NOTES: Final[str] = "marc.notes"
_FIELD_MARC_EDITION: Final[str] = "marc.edition"
_FIELD_MARC_SERIES: Final[str] = "marc.series_titles"
_FIELD_CCE_NOTES: Final[str] = "cce.notes"
_FIELD_CCE_NEW_MATTER: Final[str] = "cce.new_matter_claimed"

# Score thresholds for outcome classification. The scorer emits 100.0 on
# agreement, 25.0 on same-kind/different-number, 0.0 on whole-vs-part penalty.
_HIGH_SCORE: Final[float] = 100.0
_LOW_SCORE_MAX: Final[float] = 25.0

# A change attributing fewer than this many movable cases is flagged as not
# worth the false-positive risk (per the issue's "<~3 cases" guidance).
_MIN_MOVABLE_CASES: Final[int] = 3

_MAX_REPRESENTATIVE_ROWS: Final[int] = 20
_RAW_TEXT_TRUNCATE: Final[int] = 90
_UUID_TRUNCATE: Final[int] = 8

# Keyword surface forms for the slice's keyword-union pass (issue #82).
_KEYWORD_RE = re_compile(
    r"\bv\.?\b"
    r"|vol"
    r"|\bpt\.?\b"
    r"|part"
    r"|tome"
    r"|tomo"
    r"|\bbd\.?\b"
    r"|band"
    r"|teil"
    r"|\bt\.?\s*\d"
    r"|heft"
    r"|fasc"
    r"|livre"
    r"|libro"
    r"|complete"
    r"|collected"
    r"|selected"
    r"|\bser\.?\b"
    r"|aufl"
    r"|course\s+\d",
    IGNORECASE,
)

# Isolated part-designator token capture for roman/word attribution. Mirrors the
# kind-prefix alternation of volume._PART_NUMBER_RE but captures the raw token
# (before canonicalisation) so we can test it against the normalisers.
_PART_TOKEN_RE = re_compile(
    r"\b(?:"
    r"v(?:ol(?:ume)?)?"
    r"|pt|part"
    r"|bk|book"
    r"|t(?:omes?|omos?|omus)?"
    r"|bd|band"
    r"|tl|teil"
    r"|dl|deel"
    r"|l(?:ivres?|ibros?|iber)?"
    r"|lib"
    r"|pars"
    r"|heft"
    r")\.?\s*"
    r"(?![a-z]\.\s)"
    r"([a-z0-9]+)\b",
    IGNORECASE,
)


def _progress(message: str) -> None:
    """Emit a one-line milestone to stderr and the shared progress log."""
    print(message, file=stderr, flush=True)
    with _PROGRESS_LOG.open("a", encoding="utf-8") as handle:
        handle.write(f"{message}\n")


def _keyword_hit(value: str | None) -> bool:
    """Return ``True`` when ``value`` carries any whole/part surface form."""
    if not value:
        return False
    return _KEYWORD_RE.search(value) is not None


def _entry_keyword_hit(
    entry: VaultEntry,
    marc: MarcRecord | None,
    cce: IndexedNyplRegRecord | None,
) -> bool:
    """Return ``True`` when the note or any resolved field carries a keyword."""
    if _keyword_hit(entry.note):
        return True
    if marc is not None:
        if _keyword_hit(marc.title) or _keyword_hit(marc.extent):
            return True
        if _keyword_hit(marc.edition):
            return True
        if any(_keyword_hit(note) for note in marc.notes):
            return True
        if any(_keyword_hit(series) for series in marc.series_titles):
            return True
    if cce is not None:
        if _keyword_hit(cce.title) or _keyword_hit(cce.desc):
            return True
        if _keyword_hit(cce.new_matter_claimed):
            return True
        if any(_keyword_hit(note) for note in cce.notes):
            return True
    return False


@dataclass(frozen=True, slots=True)
class ResolvedCase:
    """A vault entry resolved to its records and current-scorer classification."""

    entry: VaultEntry
    marc: MarcRecord
    cce: IndexedNyplRegRecord
    language: str
    is_tagged: bool
    score: float
    skipped: bool
    marc_kind: str
    marc_part: str | None
    cce_kind: str
    cce_part: str | None


@dataclass(frozen=True, slots=True)
class UnreadFieldHit:
    """One field whose UNREAD text would have fired a part/whole detector."""

    field_name: str
    raw_text: str
    detector: str


@dataclass(frozen=True, slots=True)
class CaseAttribution:
    """The outcome and cause attribution for one resolved case."""

    case: ResolvedCase
    outcome: str
    cause: str
    unread_hits: tuple[UnreadFieldHit, ...]


@dataclass(slots=True)
class StudyReport:
    """Aggregate counts and rows for the characterization report."""

    slice_size: int = 0
    tagged_count: int = 0
    keyword_only_count: int = 0
    overlap_count: int = 0
    resolved: int = 0
    missing_in_pool: int = 0
    missing_in_index: int = 0
    outcomes: Counter[str] = field(default_factory=Counter)
    causes: Counter[str] = field(default_factory=Counter)
    field_marc_side: Counter[str] = field(default_factory=Counter)
    field_cce_side: Counter[str] = field(default_factory=Counter)
    attributions: list[CaseAttribution] = field(default_factory=list)
    limit: int | None = None


def _expected_score_band(entry: VaultEntry) -> str:
    """Return ``"high"`` or ``"low"`` — the gold-expected volume score band.

    A whole/part case the labeler judged ``no_match`` SHOULD score LOW (the
    scorer correctly detecting whole/part disagreement); a ``match`` case
    SHOULD score HIGH. Other verdicts (e.g. ``unsure``) have no firm
    expectation and are treated as ``high`` so a skip counts as a miss, not a
    success.
    """
    if entry.verdict == _VERDICT_NO_MATCH:
        return "low"
    return "high"


def _classify_outcome(case: ResolvedCase) -> str:
    """Return the outcome of the current scorer against the gold expectation.

    * ``correctly_scored`` — the scorer produced a NON-skipped score in the
      gold-expected band (low for ``no_match``, high for ``match``).
    * ``skipped_but_signal_exists`` — the scorer skipped (``skipped=True``);
      whether a recoverable signal exists is decided in cause attribution. A
      skip is never "correct" for a whole/part case the labeler could
      adjudicate, so it is always a non-correct outcome here.
    * ``misclassified`` — the scorer produced a non-skipped score in the WRONG
      band (e.g. a ``no_match`` whole/part case scored high, or a ``match``
      scored low).
    """
    expected = _expected_score_band(case.entry)
    if case.skipped:
        return _OUTCOME_SKIPPED_SIGNAL
    if expected == "low":
        if case.score <= _LOW_SCORE_MAX:
            return _OUTCOME_CORRECT
        return _OUTCOME_MISCLASSIFIED
    if case.score >= _HIGH_SCORE:
        return _OUTCOME_CORRECT
    return _OUTCOME_MISCLASSIFIED


def _scan_unread_fields(case: ResolvedCase) -> tuple[UnreadFieldHit, ...]:
    """Re-run part/whole detection over the fields the scorer never reads.

    The current scorer reads MARC ``title`` / ``extent`` / ``title_part_number``
    and CCE ``desc`` / ``title`` only. This re-runs ``_detect_part`` /
    ``_is_multivolume_whole`` / ``_is_collected_title`` over MARC ``notes`` /
    ``edition`` / ``series_titles`` and CCE ``notes`` / ``new_matter_claimed``,
    recording WHICH field would have fired and via WHICH detector.
    """
    hits: list[UnreadFieldHit] = []
    marc = case.marc
    cce = case.cce

    unread: list[tuple[str, str]] = []
    unread.extend((_FIELD_MARC_NOTES, note) for note in marc.notes)
    if marc.edition:
        unread.append((_FIELD_MARC_EDITION, marc.edition))
    unread.extend((_FIELD_MARC_SERIES, series) for series in marc.series_titles)
    unread.extend((_FIELD_CCE_NOTES, note) for note in cce.notes)
    if cce.new_matter_claimed:
        unread.append((_FIELD_CCE_NEW_MATTER, cce.new_matter_claimed))

    for field_name, text in unread:
        detector = _first_detector(text)
        if detector is not None:
            hits.append(
                UnreadFieldHit(field_name=field_name, raw_text=text, detector=detector)
            )
    return tuple(hits)


def _first_detector(text: str) -> str | None:
    """Return the name of the first whole/part detector that fires on ``text``."""
    if _detect_part(text) is not None:
        return "_detect_part"
    if _is_multivolume_whole(text):
        return "_is_multivolume_whole"
    if _is_collected_title(text):
        return "_is_collected_title"
    return None


def _raw_part_tokens(value: str | None) -> list[str]:
    """Return the raw (pre-canonical) part-number tokens captured from ``value``."""
    if not value:
        return []
    return [match.group(1) for match in _PART_TOKEN_RE.finditer(value)]


def _is_roman_arabic_pair(a: str, b: str) -> bool:
    """Return ``True`` when ``a`` and ``b`` denote the same integer, one roman."""
    roman_a = roman_to_arabic(a)
    roman_b = roman_to_arabic(b)
    digit_a = int(a) if a.isdigit() else None
    digit_b = int(b) if b.isdigit() else None
    if roman_a is not None and digit_b is not None and roman_a == digit_b:
        return a.lower() != b.lower()
    if roman_b is not None and digit_a is not None and roman_b == digit_a:
        return a.lower() != b.lower()
    return False


def _is_word_number_pair(a: str, b: str, language: str) -> bool:
    """Return ``True`` when one token is a number-word equal to the other digit.

    Restricted to number-words BEYOND volume.py's built-in 10-word table (which
    already canonicalises one..ten), so this attributes only misses the current
    scorer genuinely cannot resolve.
    """
    return _word_matches_digit(a, b, language) or _word_matches_digit(b, a, language)


def _word_matches_digit(word: str, digit: str, language: str) -> bool:
    """Return ``True`` when ``word`` is a number-word equal to integer ``digit``."""
    if not digit.isdigit():
        return False
    value = word_to_int(word, language)
    if value is None:
        return False
    if value <= 10:
        return False
    return value == int(digit)


def _roman_or_word_cause(case: ResolvedCase) -> str | None:
    """Return a roman/word cause when both sides are ``part`` but normalise equal.

    Both sides must classify ``part`` (so the scorer reached the
    same-kind/different-number branch and scored 25.0). The raw tokens from the
    read fields are tested pairwise: a roman-vs-arabic equality is
    ``roman_numeral_miss``; a word-vs-digit equality beyond the built-in table
    is ``word_number_miss``. Roman is checked first.
    """
    if case.marc_kind != _PART or case.cce_kind != _PART:
        return None
    if case.marc_part == case.cce_part:
        return None
    marc_tokens = _raw_part_tokens(case.marc.title) + _raw_part_tokens(case.marc.extent)
    if case.marc.title_part_number:
        marc_tokens.extend(_raw_part_tokens(case.marc.title_part_number))
    cce_tokens = _raw_part_tokens(case.cce.desc) + _raw_part_tokens(case.cce.title)
    for marc_token in marc_tokens:
        for cce_token in cce_tokens:
            if _is_roman_arabic_pair(marc_token, cce_token):
                return _CAUSE_ROMAN
    for marc_token in marc_tokens:
        for cce_token in cce_tokens:
            if _is_word_number_pair(marc_token, cce_token, case.language):
                return _CAUSE_WORD
    return None


def _attribute_cause(case: ResolvedCase) -> CaseAttribution:
    """Classify outcome, then attribute a cause to every non-correct case."""
    outcome = _classify_outcome(case)
    if outcome == _OUTCOME_CORRECT:
        return CaseAttribution(
            case=case, outcome=outcome, cause=_OUTCOME_CORRECT, unread_hits=()
        )

    unread_hits = _scan_unread_fields(case)
    if unread_hits:
        return CaseAttribution(
            case=case,
            outcome=outcome,
            cause=_CAUSE_UNREAD_FIELD,
            unread_hits=unread_hits,
        )

    roman_word = _roman_or_word_cause(case)
    if roman_word is not None:
        return CaseAttribution(
            case=case, outcome=outcome, cause=roman_word, unread_hits=()
        )

    if case.skipped and case.marc_kind == _UNKNOWN and case.cce_kind == _UNKNOWN:
        # Both sides parse to nothing and no unread field recovers a signal: a
        # designator no current regex matches, OR genuinely no signal. Eyeball
        # via the representative dump; flagged as a pattern-gap candidate.
        return CaseAttribution(
            case=case,
            outcome=outcome,
            cause=_CAUSE_PATTERN_GAP,
            unread_hits=(),
        )
    return CaseAttribution(
        case=case, outcome=outcome, cause=_CAUSE_AMBIGUOUS, unread_hits=()
    )


def run_study(
    entries: dict[tuple[str, str], VaultEntry], limit: int | None
) -> StudyReport:
    """Resolve the whole/part slice, run the scorer, and attribute every miss."""
    matching_config = _load_default_matching_config()
    report = StudyReport(limit=limit)

    tagged_keys: set[tuple[str, str]] = {
        key
        for key, entry in entries.items()
        if _WHOLE_PART_CATEGORIES.intersection(entry.categories)
    }

    # Resolve EVERY entry's MARC + CCE so the keyword pass can scan resolved
    # fields, not just the note. Bounded by the vault size (~1400), single pass.
    wanted_marc_ids = {entry.marc_control_id for entry in entries.values()}
    marc_by_id = build_marc_index(_POOL_PATH, wanted_marc_ids)
    _progress(f"resolved {len(marc_by_id)} MARC records from pool")

    selected: list[tuple[VaultEntry, MarcRecord, IndexedNyplRegRecord, bool]] = []

    with NyplIndexLookup(_INDEX_PATH) as lookup:
        idf = build_idf_table(lookup)
        author_idf = build_author_idf_table(lookup)
        publisher_idf = build_publisher_idf_table(lookup)
        _progress("idf table built; selecting slice")

        for key, entry in entries.items():
            marc = marc_by_id.get(entry.marc_control_id)
            cce = lookup.get_registration(entry.nypl_uuid)
            is_tagged = key in tagged_keys
            keyword = _entry_keyword_hit(entry, marc, cce)
            if not (is_tagged or keyword):
                continue
            report.slice_size += 1
            if is_tagged:
                report.tagged_count += 1
            if keyword and not is_tagged:
                report.keyword_only_count += 1
            if is_tagged and keyword:
                report.overlap_count += 1
            if marc is None:
                report.missing_in_pool += 1
                continue
            if cce is None:
                report.missing_in_index += 1
                continue
            selected.append((entry, marc, cce, is_tagged))

        if limit is not None:
            selected = selected[:limit]

        _progress(f"slice={report.slice_size}; resolved candidates={len(selected)}")

        for entry, marc, cce, is_tagged in selected:
            ctx = _build_context(marc, idf, author_idf, publisher_idf, matching_config)
            evidence = score_volume(marc, cce, ctx)
            marc_kind, marc_part = _classify_marc(marc)
            cce_kind, cce_part = _classify_cce(cce)
            case = ResolvedCase(
                entry=entry,
                marc=marc,
                cce=cce,
                language=ctx.language,
                is_tagged=is_tagged,
                score=evidence.score,
                skipped=evidence.skipped,
                marc_kind=marc_kind,
                marc_part=marc_part,
                cce_kind=cce_kind,
                cce_part=cce_part,
            )
            report.resolved += 1
            attribution = _attribute_cause(case)
            report.outcomes[attribution.outcome] += 1
            report.causes[attribution.cause] += 1
            for hit in attribution.unread_hits:
                if hit.field_name.startswith("marc."):
                    report.field_marc_side[hit.field_name] += 1
                else:
                    report.field_cce_side[hit.field_name] += 1
            report.attributions.append(attribution)

    _progress(
        "study done: "
        f"correct={report.outcomes[_OUTCOME_CORRECT]} "
        f"skipped={report.outcomes[_OUTCOME_SKIPPED_SIGNAL]} "
        f"misclassified={report.outcomes[_OUTCOME_MISCLASSIFIED]}"
    )
    return report


def _truncate(text: str, length: int) -> str:
    """Return ``text`` clipped to ``length`` with an ellipsis when over."""
    collapsed = " ".join(text.split())
    if len(collapsed) <= length:
        return collapsed
    return collapsed[: length - 1] + "…"


def _print_limit_warning(limit: int | None) -> None:
    """Emit a prominent SMOKE banner when a ``--limit`` was active."""
    if limit is None:
        return
    print(
        f"> ⚠️ **SMOKE RUN — `--limit {limit}` was active.** Only the "
        f"first {limit} resolved whole/part cases were scored; every count below "
        "is from a truncated set and is NOT a real finding. Re-run WITHOUT "
        "`--limit` for the production report.\n"
    )


def _movable_for_unread(report: StudyReport) -> int:
    """Return the count of cases attributed to a signal in an unread field."""
    return report.causes[_CAUSE_UNREAD_FIELD]


def _movable_for_roman(report: StudyReport) -> int:
    """Return the count of roman-numeral-miss cases."""
    return report.causes[_CAUSE_ROMAN]


def _movable_for_word(report: StudyReport) -> int:
    """Return the count of word-number-miss cases."""
    return report.causes[_CAUSE_WORD]


def _print_method(report: StudyReport) -> None:
    """Emit the Method section."""
    print("## Method\n")
    print(
        "**Slice selection.** From `current_entries(data/label_vault.jsonl)`, an "
        "entry enters the study slice if EITHER it is tagged "
        f"`{_CAT_MARC_WHOLE}` / `{_CAT_CCE_WHOLE}`, OR a whole/part surface form "
        "(case-insensitive `v.`, `vol`, `pt.`, `part`, `tome`, `tomo`, `bd.`, "
        "`band`, `teil`, `t. N`, `heft`, `fasc`, `livre`, `libro`, `complete`, "
        "`collected`, `selected`, `ser.`, `aufl`, `course N`) appears in the "
        "free-text `note` OR in any resolved record field (MARC "
        "title/extent/edition/notes/series; CCE title/desc/notes/new_matter). "
        "The union widens beyond the tagged set so keyword-only cases the "
        "labeler never tagged are still characterised.\n"
    )
    print(
        "**Resolution path.** Each entry's MARC is resolved via "
        "`build_marc_index` over the candidate pool (`data/candidates`) and its "
        "CCE via `NyplIndexLookup.get_registration` over `caches/cce.lmdb` — the "
        "same proven read-only path as `scripts/learned_scorer_heldout.py`. "
        "Entries whose MARC is absent from the pool or whose CCE is absent from "
        "the index are counted and skipped.\n"
    )
    print(
        "**Current scorer, as-is.** `score_volume` runs with a real "
        "`ScorerContext` from the production `_build_context`, so `ctx.language` "
        "is the record's true language. `_classify_marc` / `_classify_cce` are "
        "called directly to capture each side's `(cardinality, part_number)`. No "
        "`src/` code is modified.\n"
    )
    print(
        "**Outcome definitions.** The gold expectation for a whole/part case: a "
        f"`{_VERDICT_NO_MATCH}` verdict SHOULD get a LOW volume score (the scorer "
        f"correctly detecting whole/part disagreement; ≤ {_LOW_SCORE_MAX:.0f}); a "
        f"`{_VERDICT_MATCH}` verdict SHOULD get a HIGH score (≥ "
        f"{_HIGH_SCORE:.0f}). "
        f"\n\n"
        f"- **{_OUTCOME_CORRECT}** — the scorer produced a non-skipped score in "
        "the expected band.\n"
        f"- **{_OUTCOME_SKIPPED_SIGNAL}** — the scorer skipped "
        "(`skipped=True`); whether a recoverable signal exists is decided in "
        "cause attribution. A skip is never correct for an adjudicable "
        "whole/part case.\n"
        f"- **{_OUTCOME_MISCLASSIFIED}** — a non-skipped score in the WRONG "
        "band.\n"
    )
    print(
        "**Cause attribution (non-correct cases only), in priority order.**\n\n"
        f"1. **{_CAUSE_UNREAD_FIELD}** — re-running `_detect_part` / "
        "`_is_multivolume_whole` / `_is_collected_title` over the fields the "
        "scorer never reads (MARC `notes`/`edition`/`series_titles`, CCE "
        "`notes`/`new_matter_claimed`) fires on at least one. The firing field "
        "and detector are recorded. This directly validates/kills the issue's "
        "“~30% in MARC 500 notes” claim against the real resolution path.\n"
        f"2. **{_CAUSE_ROMAN}** — both sides classify `part` with different "
        "numbers, but a raw read-field token on one side is a Roman numeral "
        "equal to an Arabic token on the other (`roman_to_arabic(a) == int(b)`).\n"
        f"3. **{_CAUSE_WORD}** — analogous, a number-word vs a digit BEYOND "
        "volume.py's built-in one..ten table (`word_to_int` with the record's "
        "language).\n"
        f"4. **{_CAUSE_PATTERN_GAP}** — the scorer skipped, both sides parse to "
        "`unknown`, and no unread field recovers a signal: a human-visible "
        "designator no current regex matches (eyeball the representative dump).\n"
        f"5. **{_CAUSE_AMBIGUOUS}** — no recoverable signal anywhere; correctly "
        "beyond reach.\n"
    )


def _print_headline(report: StudyReport) -> None:
    """Emit the headline counts."""
    print("## Headline\n")
    _print_limit_warning(report.limit)
    print(
        f"- **Slice size**: {report.slice_size} "
        f"(tagged {report.tagged_count}; keyword-only {report.keyword_only_count}; "
        f"tagged∩keyword overlap {report.overlap_count})\n"
        f"- **Resolved / scored**: {report.resolved}\n"
        f"- **Unresolved**: {report.missing_in_pool} missing in pool, "
        f"{report.missing_in_index} missing in index\n"
    )
    print("| outcome | count |")
    print("|:---|---:|")
    for outcome in (_OUTCOME_CORRECT, _OUTCOME_SKIPPED_SIGNAL, _OUTCOME_MISCLASSIFIED):
        print(f"| {outcome} | {report.outcomes[outcome]} |")
    print()


def _print_cause_table(report: StudyReport) -> None:
    """Emit the per-cause ranked table (the Phase-1 priority list)."""
    print("## Cause attribution (Phase-1 priority list)\n")
    print(
        "Ranked by count over all NON-correct cases. `correctly_scored` is shown "
        "for completeness.\n"
    )
    print("| cause | count |")
    print("|:---|---:|")
    ordered = sorted(report.causes.items(), key=lambda kv: (-kv[1], kv[0]))
    for cause, count in ordered:
        print(f"| {cause} | {count} |")
    print()


def _print_field_table(report: StudyReport) -> None:
    """Emit the per-field table for ``signal_in_unread_field``."""
    print("## Signal-in-unread-field, by field\n")
    print(
        "Counts the cases where each unread field would have fired a detector "
        "(a single case can fire on multiple fields, so column sums may exceed "
        f"the {_movable_for_unread(report)} `{_CAUSE_UNREAD_FIELD}` cases). "
        "Directly validates or kills the “notes carry the signal” claim.\n"
    )
    print("| field | side | cases firing |")
    print("|:---|:---|---:|")
    for field_name in (_FIELD_MARC_NOTES, _FIELD_MARC_EDITION, _FIELD_MARC_SERIES):
        print(f"| `{field_name}` | MARC | {report.field_marc_side[field_name]} |")
    for field_name in (_FIELD_CCE_NOTES, _FIELD_CCE_NEW_MATTER):
        print(f"| `{field_name}` | CCE | {report.field_cce_side[field_name]} |")
    print()


def _print_representative_rows(report: StudyReport) -> None:
    """Emit up to ``_MAX_REPRESENTATIVE_ROWS`` representative non-correct rows.

    Prioritises ``signal_in_unread_field`` (the field + raw text carrying the
    missed signal), then roman/word, then pattern-gap candidates.
    """
    print("## Representative rows\n")
    priority = {
        _CAUSE_UNREAD_FIELD: 0,
        _CAUSE_ROMAN: 1,
        _CAUSE_WORD: 1,
        _CAUSE_PATTERN_GAP: 2,
        _CAUSE_AMBIGUOUS: 3,
    }
    candidates = [
        attribution
        for attribution in report.attributions
        if attribution.outcome != _OUTCOME_CORRECT
    ]
    candidates.sort(key=lambda a: priority.get(a.cause, 9))
    shown = candidates[:_MAX_REPRESENTATIVE_ROWS]
    if not shown:
        print("_No non-correct cases in this run._\n")
        return
    print(
        "| marc_control_id | nypl_uuid | cause | field | raw_text | score | expected |"
    )
    print("|:---|:---|:---|:---|:---|---:|:---|")
    for attribution in shown:
        case = attribution.case
        if attribution.unread_hits:
            hit = attribution.unread_hits[0]
            field_name = hit.field_name
            raw_text = _truncate(hit.raw_text, _RAW_TEXT_TRUNCATE)
        else:
            field_name = "—"
            raw_text = _truncate(_fallback_raw(case), _RAW_TEXT_TRUNCATE)
        expected = _expected_score_band(case.entry)
        uuid = case.entry.nypl_uuid[:_UUID_TRUNCATE]
        print(
            f"| {case.entry.marc_control_id} | {uuid}… | {attribution.cause} | "
            f"{field_name} | {raw_text} | {case.score:.0f} | {expected} |"
        )
    print()


def _fallback_raw(case: ResolvedCase) -> str:
    """Return a best-effort raw designator dump for rows with no unread hit."""
    parts: list[str] = []
    if case.marc.extent:
        parts.append(f"marc.extent={case.marc.extent}")
    if case.cce.desc:
        parts.append(f"cce.desc={case.cce.desc}")
    if case.marc.title_part_number:
        parts.append(f"marc.part#={case.marc.title_part_number}")
    return " | ".join(parts) if parts else "(no designator fields)"


def _print_decision(report: StudyReport) -> None:
    """Emit the programmatic Phase-1 decision section."""
    print("## Decision — which Phase-1 changes the data justifies\n")
    unread = _movable_for_unread(report)
    roman = _movable_for_roman(report)
    word = _movable_for_word(report)
    total_movable = unread + roman + word

    print(
        "Three candidate Phase-1 changes, ranked by movable-case count. A change "
        f"attributing fewer than {_MIN_MOVABLE_CASES} cases is flagged as not "
        "worth the false-positive risk it introduces.\n"
    )
    print("| change | movable cases | verdict |")
    print("|:---|---:|:---|")
    ranked = sorted(
        (
            ("A) widen consulted fields", unread),
            ("B) roman/word part-number normalisation", roman + word),
            ("C) language threading (word-number)", word),
        ),
        key=lambda kv: -kv[1],
    )
    for label, count in ranked:
        verdict = (
            "pursue" if count >= _MIN_MOVABLE_CASES else "skip (false-positive risk)"
        )
        print(f"| {label} | {count} | {verdict} |")
    print()

    if total_movable < _MIN_MOVABLE_CASES:
        print(
            f"**Recommendation: characterize and stop.** Only {total_movable} "
            "cases are movable across all three changes — below the "
            f"{_MIN_MOVABLE_CASES}-case floor. The whole/part losses are "
            "dominated by genuinely-ambiguous cases the scorer cannot recover "
            "from designators alone; Phase-1 scoring changes would add "
            "false-positive risk for no measurable top-1 gain. Close #82 at "
            "Phase 0 with this characterization.\n"
        )
        return

    print(
        f"**Recommendation: pursue the changes above the {_MIN_MOVABLE_CASES}-case "
        f"floor.** {total_movable} cases are movable in total. Per the issue's "
        "phasing, Phase 1 implements them in `volume.py` on a "
        "`phase-N-volume-designators` branch with structured-wins precedence "
        "(notes scanned last under the same guarded regex) and 100% coverage, "
        "then Phase 2 retrains the learned model and reruns the held-out eval "
        "and `pdm run regression` to confirm the weighted mean improves without "
        "a precision regression before shipping.\n"
    )
    print(
        "> Note: per the vault blind-spot finding, the labeled vault is "
        "structurally biased toward `match` pairs the matcher already surfaces; "
        "movable-case counts here are an UPPER BOUND on top-1 flips, not a "
        "prediction. Phase 2's per-MARC diagnostic is the real gate.\n"
    )


def _print_report(report: StudyReport) -> None:
    """Emit the full markdown report to stdout."""
    print("# Volume signal characterization (whole/part) — 2026-06-13\n")
    _print_limit_warning(report.limit)
    print(
        "Issue #82, Phase 0. Runs the CURRENT `volume.compat` scorer AS-IS over "
        "the whole/part vault slice and attributes every non-correct case to a "
        "nameable CAUSE and FIELD, so Phase 1 builds only the changes the data "
        "justifies. No `src/` code is modified; the vault is read-only.\n"
    )
    _print_method(report)
    _print_headline(report)
    _print_cause_table(report)
    _print_field_table(report)
    _print_representative_rows(report)
    _print_decision(report)


def _parse_limit() -> int | None:
    """Parse the optional ``--limit N`` smoke flag from argv."""
    parser = ArgumentParser(
        description="Phase-0 volume signal characterization (issue #82)."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        metavar="N",
        help="cap the resolved whole/part cases scored (smoke; flagged)",
    )
    args = parser.parse_args()
    if args.limit is not None and args.limit <= 0:
        parser.error("--limit must be a positive integer")
    limit: int | None = args.limit
    return limit


def main() -> None:
    """Run the characterization study and print the markdown report to stdout."""
    limit = _parse_limit()
    _progress(f"volume_signal_characterization started (limit={limit})")
    entries = current_entries(_VAULT_PATH)
    report = run_study(entries, limit)
    _print_report(report)
    _progress("report written")


if __name__ == "__main__":
    main()
