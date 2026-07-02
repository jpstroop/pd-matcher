"""Renewal-matcher training + HONEST held-out evaluation (GitHub #45).

Answers one question: does the harvested MARC↔renewal training data produce a
working renewal matcher, and does a *trained* model beat the untrained
weighted-mean baseline on an honest, leakage-controlled test?

**v2 (A vs B).** Two feature sets are compared. Feature set **A** is the prior
8-dim per-scorer vector (title / author / claimant / year, each a normalized
reading plus a present flag). Feature set **B** appends three domain features
read straight off the renewal record: (1) the ``oreg`` registration class as
book / periodical / drama indicators; (2) the statutory claimant-class as
author / estate / proprietor indicators (proprietor is the mismatch-risk flag);
(3) a *class-conditioned* author name-match — the claimant-vs-MARC-author
name similarity, active only when the renewal is claimed as author, since an
estate or proprietor renews under a different name and must not be penalized.
The eval reports grouped-CV AUC for A and B, then the honest external vault AUC
and P/R for the baseline, trained-A, and trained-B arms, plus trained-B's
coefficients — the test of whether the domain features let a trained arm beat
the untrained weighted-mean where it counts.

The eval is STANDALONE. It does not touch the production combiner
(``match/combiners/features.py``); it only *reuses* the production renewal
scorers to turn each ``(MARC, renewal)`` pair into a per-scorer feature vector,
exactly as ``score_renewal`` in ``build_renewal_queue`` does.

Two datasets:

* **Harvested** (``data/harvested_renewal_pairs.jsonl``): 220 verified-by-
  transitivity positives + 220 same-MARC hard-negative look-alikes. Self-
  contained — MARC + renewal fields reconstructed straight from the JSONL, no
  pool lookup. Used for grouped-by-MARC cross-validation.
* **Vault renewal entries** (``data/training/label_vault.jsonl``,
  ``match_source == "renewal"``): the ~117 match / 14 no_match HUMAN-labeled
  renewal verdicts. READ-ONLY. Their MARC is resolved from the candidate pool
  (``data/candidates``) via ``build_marc_index`` — the same way the harvest tool
  does it, NOT from ``marc.xml`` — and their renewal from the LMDB index by
  ``entry_id``. This never-trained-on set is the honest external test.

Baseline = the weighted-mean renewal combiner (untrained), the exact scorer
``build-renewal-queue`` uses. Trained = a well-regularized logistic regression
(tiny N ~= 440 harvested + ~120 vault; a linear L2 model resists overfitting far
better than a tree ensemble on 8 features and this few rows).

No leakage: harvested CV is grouped by MARC control id (a MARC and its own hard
negatives never straddle the train/test split). The vault set shares no rows
with training and is scored only after the model is frozen.

Outputs numbers to stdout for transcription into
``docs/findings/renewal_matcher_eval_2026-07-01.md``. Writes nothing under
``data/``. Optionally persists the trained model under ``caches/`` (gitignored).
"""

from collections.abc import Iterator
from datetime import date
from logging import INFO
from logging import basicConfig
from logging import getLogger
from pathlib import Path
from re import split as re_split
from statistics import mean
from statistics import pstdev

from msgspec.json import decode as json_decode
from numpy import asarray
from numpy import ndarray
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from pd_groundtruth.label_vault import iter_entries
from pd_groundtruth.vault_pair_resolver import build_marc_index
from pd_matcher.cli import _AUTHOR_IDF_CACHE_NAME
from pd_matcher.cli import _IDF_CACHE_NAME
from pd_matcher.cli import _PUBLISHER_IDF_CACHE_NAME
from pd_matcher.cli import _load_calibrator
from pd_matcher.cli import _load_default_matching_config
from pd_matcher.index.lookup import NyplIndexLookup
from pd_matcher.match.combiners import build_combiner
from pd_matcher.match.combiners.base import Combiner
from pd_matcher.match.combiners.calibrator import PlattCalibrator
from pd_matcher.match.combiners.calibrator import calibrate
from pd_matcher.match.evidence import Evidence
from pd_matcher.match.idf import load_or_build_author_idf
from pd_matcher.match.idf import load_or_build_idf
from pd_matcher.match.idf import load_or_build_publisher_idf
from pd_matcher.match.pipeline import _build_context
from pd_matcher.match.scorers.context import ScorerContext
from pd_matcher.match.scorers.name import score_author
from pd_matcher.match.scorers.name import score_publisher
from pd_matcher.match.scorers.title import score_title
from pd_matcher.match.scorers.year import score_year
from pd_matcher.models import MarcRecord
from pd_matcher.models import NyplRenRecord
from pd_matcher.normalize.registration_numbers import reg_class

_LOGGER = getLogger("renewal_matcher_eval")

_CACHES = Path("caches")
_INDEX = _CACHES / "cce.lmdb"
_POOL = Path("data/candidates")
_HARVEST = Path("data/harvested_renewal_pairs.jsonl")
_VAULT = Path("data/training/label_vault.jsonl")
_PROGRESS = Path("/tmp/agent-progress.log")

_FEATURE_NAMES_A: tuple[str, ...] = (
    "title_norm",
    "title_present",
    "author_norm",
    "author_present",
    "claimants_norm",
    "claimants_present",
    "year_norm",
    "year_present",
)

_FEATURE_NAMES_EXTRA: tuple[str, ...] = (
    "oreg_book",
    "oreg_periodical",
    "oreg_drama",
    "claim_author",
    "claim_estate",
    "claim_proprietor",
    "name_match_cond",
    "name_expected",
)

_FEATURE_NAMES_B: tuple[str, ...] = _FEATURE_NAMES_A + _FEATURE_NAMES_EXTRA

_N_FEATURES_A: int = len(_FEATURE_NAMES_A)
_N_FEATURES_B: int = len(_FEATURE_NAMES_B)

_OREG_BOOK: frozenset[str] = frozenset({"A", "AA", "AF", "AI", "AFO", "AIO"})
_OREG_PERIODICAL: frozenset[str] = frozenset({"B", "BB"})
_OREG_DRAMA: frozenset[str] = frozenset({"D", "DF", "DP"})

_CODE_AUTHOR: frozenset[str] = frozenset({"A"})
_CODE_ESTATE: frozenset[str] = frozenset({"W", "C", "E", "NK"})
_CODE_PROPRIETOR: frozenset[str] = frozenset({"PWH", "PPW", "PCW"})

_CV_SPLITS: int = 5
_LOGREG_C: float = 1.0
_RANDOM_STATE: int = 0


class Sample:
    """One labeled ``(MARC, renewal)`` pair: features, baseline, label, group.

    ``features`` is the full feature-set-B vector (the 8-dim per-scorer block
    followed by the 8-dim domain block); feature set A is its first
    :data:`_N_FEATURES_A` columns. ``baseline`` is the weighted-mean combiner's
    calibrated score for the pair; ``label`` is 1 for a true match, 0 otherwise;
    ``group`` is the MARC control id used to keep a record and its hard negatives
    on the same side of a CV split.
    """

    __slots__ = ("baseline", "features", "group", "label")

    def __init__(self, features: list[float], baseline: float, label: int, group: str) -> None:
        self.features = features
        self.baseline = baseline
        self.label = label
        self.group = group


def _progress(message: str) -> None:
    """Append a timestamped milestone to the shared progress log."""
    with _PROGRESS.open("a", encoding="utf-8") as handle:
        handle.write(f"{date.today().isoformat()} renewal_eval: {message}\n")
    _LOGGER.info(message)


def _renewal_evidence(
    marc: MarcRecord, renewal: NyplRenRecord, ctx: ScorerContext
) -> tuple[Evidence, Evidence, Evidence, Evidence]:
    """Return the (title, author, claimants, year) Evidence for one pairing.

    Mirrors ``pd_groundtruth.build_renewal_queue.score_renewal`` exactly: the
    renewal's title/author are scored against both MARC title fields and both
    MARC author fields (best non-skipped kept), claimants against the MARC
    publisher, and the renewal's ``odat`` year against the MARC year.
    """
    title_candidates = tuple(
        score_title(value, renewal.title, ctx) for value in (marc.title, marc.title_main) if value
    ) or (score_title(marc.title, renewal.title, ctx),)
    author_candidates = tuple(
        score_author(value, renewal.author, ctx)
        for value in (marc.main_author, marc.statement_of_responsibility)
        if value
    ) or (score_author(marc.main_author, renewal.author, ctx),)
    title_evidence = _best(title_candidates)
    author_evidence = _best(author_candidates)
    claimants_evidence = score_publisher(marc.publisher, renewal.claimants, ctx)
    renewal_year = renewal.odat.year if renewal.odat is not None else None
    year_evidence = score_year(marc.publication_year, renewal_year, ctx)
    return title_evidence, author_evidence, claimants_evidence, year_evidence


def _best(candidates: tuple[Evidence, ...]) -> Evidence:
    """Return the highest-scoring non-skipped Evidence, else the first."""
    best = candidates[0]
    best_score = best.score if not best.skipped else -1.0
    for evidence in candidates[1:]:
        current = evidence.score if not evidence.skipped else -1.0
        if current > best_score:
            best_score = current
            best = evidence
    return best


def _feature_vector(evidence: tuple[Evidence, Evidence, Evidence, Evidence]) -> list[float]:
    """Project four Evidence into the 8-dim ``[norm, present]`` feature vector."""
    vector: list[float] = []
    for item in evidence:
        vector.append(item.normalized)
        vector.append(0.0 if item.skipped else 1.0)
    return vector


def _parse_claimants(claimants: str | None) -> tuple[tuple[str, str], ...]:
    """Return ``(name, statutory_code)`` pairs parsed from a claimants string.

    NYPL transcribes each renewal claimant as ``Name|CODE``, joining multiple
    claimants with ``||``; a minority use ``;`` separators or a trailing
    ``Name (CODE)``. The statutory code names the renewal-right class — ``A``
    author, ``W`` widow/er, ``C`` child, ``E`` executor, ``NK`` next-of-kin,
    ``PWH``/``PPW``/``PCW`` proprietor / work-for-hire. Parts without a
    recognizable trailing code yield an empty code.
    """
    if not claimants:
        return ()
    pairs: list[tuple[str, str]] = []
    for raw_part in re_split(r"\|\||;", claimants):
        part = raw_part.strip()
        if not part:
            continue
        head, _, tail = part.rpartition("|")
        code = tail.strip()
        if head and code.isalpha() and code.isupper():
            pairs.append((head.strip(), code))
            continue
        if part.endswith(")") and "(" in part:
            name, _, bracket = part.rpartition("(")
            inner = bracket[:-1].strip()
            if inner.isalpha() and inner.isupper():
                pairs.append((name.strip(), inner))
                continue
        pairs.append((part, ""))
    return tuple(pairs)


def _oreg_indicators(oreg: str | None) -> tuple[float, float, float]:
    """Return ``(is_book, is_periodical, is_drama)`` for a renewal ``oreg``.

    Uses :func:`reg_class` to read the leading registration class token, then
    buckets it into the three families that survive the in-scope filter; an
    unrecognized class produces all-zero (the implicit "else" bucket).
    """
    cls = reg_class(oreg)
    return (
        1.0 if cls in _OREG_BOOK else 0.0,
        1.0 if cls in _OREG_PERIODICAL else 0.0,
        1.0 if cls in _OREG_DRAMA else 0.0,
    )


def _claimant_indicators(pairs: tuple[tuple[str, str], ...]) -> tuple[float, float, float]:
    """Return ``(is_author, is_estate, is_proprietor)`` over parsed claimants.

    Any author-coded claimant sets ``is_author``; any of widow/child/executor/
    next-of-kin sets ``is_estate``; any proprietor / work-for-hire code sets
    ``is_proprietor`` — the risk flag, since a proprietor renewal need not share
    a name with the MARC author.
    """
    codes = {code for _, code in pairs}
    return (
        1.0 if codes & _CODE_AUTHOR else 0.0,
        1.0 if codes & _CODE_ESTATE else 0.0,
        1.0 if codes & _CODE_PROPRIETOR else 0.0,
    )


def _name_match_features(
    marc: MarcRecord,
    pairs: tuple[tuple[str, str], ...],
    ctx: ScorerContext,
) -> tuple[float, float]:
    """Return ``(name_match_cond, name_expected)`` for the class-conditioned name.

    The name-similarity between the renewal claimant and the MARC author is only
    meaningful when the renewal right is claimed *as author* (code ``A``): an
    estate or proprietor renews under a different name, so a name mismatch there
    is not evidence against a true match. When author is expected, the first
    author-coded claimant name is scored against the MARC author with the
    production author scorer and the normalized reading returned; otherwise both
    features are zero.
    """
    author_name: str | None = None
    for name, code in pairs:
        if code in _CODE_AUTHOR:
            author_name = name
            break
    if author_name is None:
        return 0.0, 0.0
    evidence = score_author(marc.main_author, author_name, ctx)
    return evidence.normalized, 1.0


def _extra_features(marc: MarcRecord, renewal: NyplRenRecord, ctx: ScorerContext) -> list[float]:
    """Return the 8-dim domain feature block appended for feature set B."""
    pairs = _parse_claimants(renewal.claimants)
    oreg_book, oreg_periodical, oreg_drama = _oreg_indicators(renewal.oreg)
    claim_author, claim_estate, claim_proprietor = _claimant_indicators(pairs)
    name_match_cond, name_expected = _name_match_features(marc, pairs, ctx)
    return [
        oreg_book,
        oreg_periodical,
        oreg_drama,
        claim_author,
        claim_estate,
        claim_proprietor,
        name_match_cond,
        name_expected,
    ]


def _baseline_score(
    evidence: tuple[Evidence, Evidence, Evidence, Evidence],
    combiner: Combiner,
    calibrator: PlattCalibrator | None,
) -> float:
    """Return the weighted-mean combiner's calibrated score for the pairing."""
    combined = combiner.combine(evidence)
    if calibrator is not None:
        return calibrate(combined.raw, calibrator)
    return combined.calibrated


class _ContextCache:
    """Per-MARC ScorerContext cache (context build is stemmer/stopword heavy)."""

    __slots__ = ("_author_idf", "_cache", "_config", "_idf", "_publisher_idf")

    def __init__(self) -> None:
        self._idf = load_or_build_idf(_CACHES / _IDF_CACHE_NAME, lambda: NyplIndexLookup(_INDEX))
        self._author_idf = load_or_build_author_idf(
            _CACHES / _AUTHOR_IDF_CACHE_NAME, lambda: NyplIndexLookup(_INDEX)
        )
        self._publisher_idf = load_or_build_publisher_idf(
            _CACHES / _PUBLISHER_IDF_CACHE_NAME, lambda: NyplIndexLookup(_INDEX)
        )
        self._config = _load_default_matching_config()
        self._cache: dict[str, ScorerContext] = {}

    @property
    def config_year_window(self) -> int:
        """The active retrieval year window (documented in the findings)."""
        return self._config.year_window

    def context_for(self, marc: MarcRecord) -> ScorerContext:
        """Return (and memoize) the ScorerContext for ``marc``."""
        ctx = self._cache.get(marc.control_id)
        if ctx is None:
            ctx = _build_context(
                marc, self._idf, self._author_idf, self._publisher_idf, self._config
            )
            self._cache[marc.control_id] = ctx
        return ctx


def _harvested_records() -> Iterator[dict[str, object]]:
    """Yield each harvested JSONL row as a decoded dict."""
    with _HARVEST.open("rb") as handle:
        for line in handle:
            if line.strip():
                yield json_decode(line)


def _marc_from_harvest(row: dict[str, object]) -> MarcRecord:
    """Reconstruct the MARC fields the renewal scorers read from a harvested row.

    The harvest stores a single ``marc_title`` / ``marc_author`` (not the
    distinct ``title_main`` / ``statement_of_responsibility`` sub-fields), so
    both title fields and the main author are seeded from those single values;
    ``_renewal_evidence``'s best-of collapses the duplicate title candidates.
    """
    title = str(row["marc_title"] or "")
    author_raw = row["marc_author"]
    author = str(author_raw) if author_raw is not None else None
    publisher_raw = row["marc_publisher"]
    year_raw = row["marc_year"]
    return MarcRecord(
        control_id=str(row["marc_control_id"]),
        title=title,
        title_main=title,
        main_author=author,
        publisher=str(publisher_raw) if publisher_raw is not None else None,
        publication_year=_as_int(year_raw) if year_raw is not None else None,
    )


def _as_int(value: object) -> int:
    """Coerce a decoded JSON numeric to ``int`` with a typed guard."""
    if isinstance(value, (int, float)):
        return int(value)
    raise TypeError(f"expected numeric year, got {type(value).__name__}")


def _renewal_from_harvest(row: dict[str, object]) -> NyplRenRecord:
    """Reconstruct the renewal fields the scorers read from a harvested row."""
    odat_raw = row["renewal_odat"]
    odat = date.fromisoformat(str(odat_raw)) if odat_raw is not None else None
    return NyplRenRecord(
        id=str(row["renewal_id"]),
        entry_id=str(row["renewal_entry_id"]),
        oreg=_opt_str(row["renewal_oreg"]),
        odat=odat,
        author=_opt_str(row["renewal_author"]),
        title=_opt_str(row["renewal_title"]),
        claimants=_opt_str(row["renewal_claimants"]),
    )


def _opt_str(value: object) -> str | None:
    """Coerce a JSON value to ``str | None``."""
    return None if value is None else str(value)


def build_harvested_samples(
    context_cache: _ContextCache, combiner: Combiner, calibrator: PlattCalibrator | None
) -> tuple[list[Sample], list[float]]:
    """Return (samples, stored_scores) for every harvested row.

    ``stored_scores`` is the ``score`` column the harvest wrote (the true
    weighted-mean baseline computed against the *full* pool MARC), returned
    alongside so the recomputed baseline can be validated against it — a
    fidelity check on the field-reconstruction shortcut.
    """
    samples: list[Sample] = []
    stored: list[float] = []
    for row in _harvested_records():
        marc = _marc_from_harvest(row)
        renewal = _renewal_from_harvest(row)
        ctx = context_cache.context_for(marc)
        evidence = _renewal_evidence(marc, renewal, ctx)
        features = _feature_vector(evidence) + _extra_features(marc, renewal, ctx)
        baseline = _baseline_score(evidence, combiner, calibrator)
        label = 1 if row["label"] == "match" else 0
        samples.append(Sample(features, baseline, label, marc.control_id))
        stored.append(_as_float(row["score"]))
    return samples, stored


def _as_float(value: object) -> float:
    """Coerce a decoded JSON numeric to ``float`` with a typed guard."""
    if isinstance(value, (int, float)):
        return float(value)
    raise TypeError(f"expected numeric score, got {type(value).__name__}")


def build_vault_samples(
    context_cache: _ContextCache, combiner: Combiner, calibrator: PlattCalibrator | None
) -> tuple[list[Sample], dict[str, int]]:
    """Resolve and score the human-labeled renewal-pathway vault entries.

    Returns the resolved samples plus a counts dict recording how many entries
    were considered, dropped as ``unsure``, missing from the pool, or missing
    from the renewal index — the honest denominator for the external test.
    """
    entries = [entry for entry in iter_entries(_VAULT) if entry.match_source == "renewal"]
    considered = [entry for entry in entries if entry.verdict in {"match", "no_match"}]
    dropped_unsure = len(entries) - len(considered)
    wanted = {entry.marc_control_id for entry in considered}
    _progress(f"vault: building MARC index for {len(wanted)} control ids (scans pool)")
    marc_by_id = build_marc_index(_POOL, wanted)
    _progress(f"vault: resolved {len(marc_by_id)} MARC records from pool")
    samples: list[Sample] = []
    missing_pool = 0
    missing_index = 0
    with NyplIndexLookup(_INDEX) as lookup:
        for entry in considered:
            marc = marc_by_id.get(entry.marc_control_id)
            if marc is None:
                missing_pool += 1
                continue
            renewal = lookup.get_renewal(entry.nypl_uuid)
            if renewal is None:
                missing_index += 1
                continue
            ctx = context_cache.context_for(marc)
            evidence = _renewal_evidence(marc, renewal, ctx)
            features = _feature_vector(evidence) + _extra_features(marc, renewal, ctx)
            baseline = _baseline_score(evidence, combiner, calibrator)
            label = 1 if entry.verdict == "match" else 0
            samples.append(Sample(features, baseline, label, entry.marc_control_id))
    counts = {
        "renewal_entries": len(entries),
        "dropped_unsure": dropped_unsure,
        "considered": len(considered),
        "missing_pool": missing_pool,
        "missing_index": missing_index,
        "resolved": len(samples),
        "resolved_match": sum(s.label for s in samples),
        "resolved_no_match": sum(1 for s in samples if s.label == 0),
    }
    return samples, counts


def _matrix(samples: list[Sample], n_features: int) -> tuple[ndarray, ndarray, list[str], ndarray]:
    """Return (X, y, groups, baseline) numpy arrays for a sample list.

    ``X`` is sliced to the first ``n_features`` columns so the same stored 16-dim
    feature vector serves both feature set A (:data:`_N_FEATURES_A`) and feature
    set B (:data:`_N_FEATURES_B`) without rescoring.
    """
    features = asarray([s.features[:n_features] for s in samples], dtype=float)
    labels = asarray([s.label for s in samples], dtype=int)
    groups = [s.group for s in samples]
    baseline = asarray([s.baseline for s in samples], dtype=float)
    return features, labels, groups, baseline


def _new_model() -> Pipeline:
    """Return a standardized, L2-regularized logistic-regression pipeline."""
    return Pipeline(
        steps=[
            ("scale", StandardScaler()),
            (
                "logreg",
                LogisticRegression(C=_LOGREG_C, max_iter=1000, random_state=_RANDOM_STATE),
            ),
        ]
    )


def grouped_cv(samples: list[Sample], n_features: int) -> tuple[list[float], list[float]]:
    """Grouped-by-MARC k-fold: return per-fold (trained AUC, baseline AUC).

    The baseline AUC is computed on the *same* held-out fold so the two arms
    are compared on identical rows; the baseline needs no training, so only the
    fold membership varies it. ``n_features`` selects feature set A or B; the
    fold assignment is identical across feature sets (same rows, same groups,
    same splitter seed), so A-vs-B is a like-for-like comparison per fold.
    """
    features, labels, groups, baseline = _matrix(samples, n_features)
    splitter = GroupKFold(n_splits=_CV_SPLITS)
    trained_aucs: list[float] = []
    baseline_aucs: list[float] = []
    for train_idx, test_idx in splitter.split(features, labels, groups):
        model = _new_model()
        model.fit(features[train_idx], labels[train_idx])
        proba = model.predict_proba(features[test_idx])[:, 1]
        trained_aucs.append(float(roc_auc_score(labels[test_idx], proba)))
        baseline_aucs.append(float(roc_auc_score(labels[test_idx], baseline[test_idx])))
    return trained_aucs, baseline_aucs


def _best_f1_threshold(scores: ndarray, labels: ndarray) -> float:
    """Return the score threshold maximizing F1 on ``(scores, labels)``.

    Chosen on the harvested (training) data only, then applied unchanged to the
    external vault test — a legitimate held-out thresholding protocol.
    """
    order = sorted(set(scores.tolist()))
    best_threshold = 0.5
    best_f1 = -1.0
    for threshold in order:
        predicted = scores >= threshold
        tp = int(((predicted == 1) & (labels == 1)).sum())
        fp = int(((predicted == 1) & (labels == 0)).sum())
        fn = int(((predicted == 0) & (labels == 1)).sum())
        denominator = 2 * tp + fp + fn
        f1 = (2 * tp) / denominator if denominator > 0 else 0.0
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    return best_threshold


def _pr(scores: ndarray, labels: ndarray, threshold: float) -> dict[str, float]:
    """Return precision/recall/confusion at ``threshold`` for the positive class."""
    predicted = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(labels, predicted, labels=[0, 1]).ravel()
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    return {
        "threshold": float(threshold),
        "precision": float(precision),
        "recall": float(recall),
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "tn": float(tn),
    }


def _pearson(left: list[float], right: ndarray) -> float:
    """Return the Pearson correlation between a python list and a numpy array."""
    left_arr = asarray(left, dtype=float)
    if float(left_arr.std()) == 0.0 or float(right.std()) == 0.0:
        return 0.0
    return float(
        ((left_arr - left_arr.mean()) * (right - right.mean())).mean()
        / (left_arr.std() * right.std())
    )


class ArmResult:
    """One feature set's trained-arm results: CV AUCs, vault AUC/PR, coefficients."""

    __slots__ = (
        "coefs",
        "cv_baseline",
        "cv_trained",
        "intercept",
        "resub_auc",
        "vault_auc",
        "vault_pr",
    )

    def __init__(
        self,
        cv_trained: list[float],
        cv_baseline: list[float],
        resub_auc: float,
        vault_auc: float,
        vault_pr: dict[str, float],
        coefs: dict[str, float],
        intercept: float,
    ) -> None:
        self.cv_trained = cv_trained
        self.cv_baseline = cv_baseline
        self.resub_auc = resub_auc
        self.vault_auc = vault_auc
        self.vault_pr = vault_pr
        self.coefs = coefs
        self.intercept = intercept


def _run_arm(
    n_features: int,
    feature_names: tuple[str, ...],
    harvested: list[Sample],
    vault: list[Sample],
) -> ArmResult:
    """Train on the full harvested set, evaluate on the vault, for one feature set.

    Runs grouped-by-MARC CV on the harvested rows, then freezes a model on all
    harvested rows and scores the never-trained-on vault. The F1 threshold is
    chosen on the harvested (training) scores and applied unchanged to the vault.
    """
    cv_trained, cv_baseline = grouped_cv(harvested, n_features)
    h_features, h_labels, _, _ = _matrix(harvested, n_features)
    v_features, v_labels, _, _ = _matrix(vault, n_features)
    model = _new_model()
    model.fit(h_features, h_labels)
    h_scores = model.predict_proba(h_features)[:, 1]
    resub_auc = float(roc_auc_score(h_labels, h_scores))
    v_scores = model.predict_proba(v_features)[:, 1]
    vault_auc = float(roc_auc_score(v_labels, v_scores))
    threshold = _best_f1_threshold(h_scores, h_labels)
    vault_pr = _pr(v_scores, v_labels, threshold)
    logreg: LogisticRegression = model.named_steps["logreg"]
    coefs = dict(zip(feature_names, logreg.coef_[0].tolist(), strict=True))
    return ArmResult(
        cv_trained,
        cv_baseline,
        resub_auc,
        vault_auc,
        vault_pr,
        coefs,
        float(logreg.intercept_[0]),
    )


def _print_pr(label: str, pr: dict[str, float]) -> None:
    """Print a labeled precision/recall/confusion line at its threshold."""
    print(
        f"{label} P/R @ thr={pr['threshold']:.3f}  "
        f"P={pr['precision']:.3f} R={pr['recall']:.3f} "
        f"tp={int(pr['tp'])} fp={int(pr['fp'])} fn={int(pr['fn'])} tn={int(pr['tn'])}"
    )


def _print_arm(name: str, result: ArmResult) -> None:
    """Print a feature set's CV, vault, and coefficient block."""
    print(f"\n--- Feature set {name} ---")
    folds = [round(a, 4) for a in result.cv_trained]
    print(f"  grouped {_CV_SPLITS}-fold CV trained AUC = {folds}")
    print(
        f"  trained AUC mean±sd       = "
        f"{mean(result.cv_trained):.4f} ± {pstdev(result.cv_trained):.4f}"
    )
    print(f"  resubstitution AUC        = {result.resub_auc:.4f}  (overfit check vs CV)")
    print(f"  vault trained AUC         = {result.vault_auc:.4f}")
    _print_pr("  vault trained", result.vault_pr)
    print("  coefficients (standardized):")
    for feature_name, value in result.coefs.items():
        print(f"    {feature_name:18s} = {value:+.4f}")
    print(f"    {'intercept':18s} = {result.intercept:+.4f}")


def main() -> None:
    """Run the A-vs-B eval and print a transcribable report to stdout."""
    basicConfig(level=INFO, format="%(asctime)s %(levelname)s %(message)s")
    _progress("start (v2: A vs B feature sets)")
    config = _load_default_matching_config()
    combiner = build_combiner(config, learned_model_dir=None)
    calibrator = _load_calibrator(_CACHES)
    context_cache = _ContextCache()

    _progress("scoring harvested set")
    harvested, stored = build_harvested_samples(context_cache, combiner, calibrator)
    _, h_labels, _, h_baseline = _matrix(harvested, _N_FEATURES_A)
    fidelity = _pearson(stored, h_baseline)
    _progress(f"harvested scored: {len(harvested)} rows, fidelity r={fidelity:.4f}")

    _progress("scoring vault external test")
    vault, counts = build_vault_samples(context_cache, combiner, calibrator)
    _, v_labels, _, v_baseline = _matrix(vault, _N_FEATURES_A)

    baseline_cv = grouped_cv(harvested, _N_FEATURES_A)[1]
    vault_baseline_auc = float(roc_auc_score(v_labels, v_baseline))
    baseline_threshold = _best_f1_threshold(h_baseline, h_labels)
    vault_baseline_pr = _pr(v_baseline, v_labels, baseline_threshold)

    _progress("running arm A")
    arm_a = _run_arm(_N_FEATURES_A, _FEATURE_NAMES_A, harvested, vault)
    _progress("running arm B")
    arm_b = _run_arm(_N_FEATURES_B, _FEATURE_NAMES_B, harvested, vault)

    print("\n============ RENEWAL MATCHER EVAL v2 (#45) — A vs B ============")
    print(f"year_window (retrieval) = {context_cache.config_year_window}")
    print(f"calibrator present      = {calibrator is not None}")
    print("\n--- Harvested set ---")
    print(
        f"rows                    = {len(harvested)}  "
        f"(pos={int(h_labels.sum())}, neg={int((h_labels == 0).sum())})"
    )
    print(f"unique MARC groups      = {len({s.group for s in harvested})}")
    print(
        f"baseline-fidelity r     = {fidelity:.4f}  "
        "(recomputed vs stored 'score'; <1 from field reconstruction)"
    )

    print("\n--- Vault external test (human-labeled, never trained on) ---")
    print(f"renewal entries         = {counts['renewal_entries']}")
    print(f"dropped 'unsure'        = {counts['dropped_unsure']}")
    print(f"considered              = {counts['considered']}")
    print(f"missing in pool         = {counts['missing_pool']}")
    print(f"missing in index        = {counts['missing_index']}")
    print(
        f"RESOLVED                = {counts['resolved']}  "
        f"(match={counts['resolved_match']}, no_match={counts['resolved_no_match']})"
    )

    print("\n--- Baseline (untrained weighted-mean) ---")
    print(f"  harvested CV baseline AUC = {mean(baseline_cv):.4f} ± {pstdev(baseline_cv):.4f}")
    print(f"  vault baseline AUC        = {vault_baseline_auc:.4f}")
    _print_pr("  vault baseline", vault_baseline_pr)

    _print_arm("A (prior: title/author/claimant/year)", arm_a)
    _print_arm("B (A + oreg-class + claimant-class + name-cond)", arm_b)

    print("\n--- Verdict inputs (vault AUC) ---")
    print(f"  baseline = {vault_baseline_auc:.4f}")
    print(f"  trained A = {arm_a.vault_auc:.4f}")
    print(f"  trained B = {arm_b.vault_auc:.4f}")
    print(f"  B beats baseline on vault = {arm_b.vault_auc > vault_baseline_auc}")
    print(f"  B beats A on vault        = {arm_b.vault_auc > arm_a.vault_auc}")
    print("===============================================================\n")
    _progress(
        f"done: vault baseline={vault_baseline_auc:.3f} "
        f"trainedA={arm_a.vault_auc:.3f} trainedB={arm_b.vault_auc:.3f}"
    )


if __name__ == "__main__":
    main()
