"""Phase-3b gate: how often could the renewal pathway recover a match the
registration pathway misses?

GitHub #45. This is a read-only DIRECTIONAL measurement, not a pipeline
change. It quantifies the upper bound on what a renewal-side matcher could
add before any integration work is justified:

1. Take an in-scope (moving-wall monograph BOOK) sample from the local
   fixture ``data/candidate_marc_file.marcxml`` using the production
   eligibility predicate (:func:`pd_groundtruth.filters.is_eligible`).
2. Run each record through the production REGISTRATION pathway exactly as the
   ``match`` CLI does — inverted-index retrieval (:meth:`candidates_for`) +
   weighted-mean scoring + the configured calibrator and floor
   (:func:`pd_matcher.match.pipeline.match_record`) — and split the sample
   into "has a confident registration match" vs "no registration match".
3. For the NO-registration-match set, retrieve renewal candidates directly
   with the new :meth:`candidates_for_renewal`, score each renewal's
   title/author/claimants + original-registration year with the SAME
   weighted-mean combiner, calibrator, and floor, and count how many records
   gain a confident renewal-only match.

Reported numbers: N (in-scope sampled), X (no registration match), Y (of
those, a confident renewal-only match), and Y as a percentage of N — the
directional ceiling for the renewal pathway.

Caveat: this is a FIXTURE-SAMPLE proxy. ``data/candidate_marc_file.marcxml``
is a raw test fixture (~18% in-scope), not the full catalog; the full-corpus
version of this gate would run over ``data/corpus.marcxml`` and compare
against the production ``pairs90.jsonl`` when available. The renewal arm
reuses the registration-side IDF tables so the two arms score on an identical
scale; a renewal-specific IDF could shift the absolute number.
"""

from collections.abc import Iterator
from pathlib import Path
from sys import stderr
from time import perf_counter

from lxml.etree import iterparse

from pd_groundtruth.acquire import default_min_year
from pd_groundtruth.filters import is_eligible
from pd_matcher.cli import _AUTHOR_IDF_CACHE_NAME
from pd_matcher.cli import _IDF_CACHE_NAME
from pd_matcher.cli import _PUBLISHER_IDF_CACHE_NAME
from pd_matcher.cli import _load_calibrator
from pd_matcher.cli import _load_default_matching_config
from pd_matcher.cli import _load_default_pairing_config
from pd_matcher.cli import _override_matching_config
from pd_matcher.match.combiners import build_combiner
from pd_matcher.match.combiners.base import CombinedScore
from pd_matcher.match.combiners.base import Combiner
from pd_matcher.match.combiners.calibrator import PlattCalibrator
from pd_matcher.match.combiners.calibrator import calibrate
from pd_matcher.match.evidence import Evidence
from pd_matcher.match.idf import IdfTable
from pd_matcher.match.idf import load_or_build_author_idf
from pd_matcher.match.idf import load_or_build_idf
from pd_matcher.match.idf import load_or_build_publisher_idf
from pd_matcher.match.pairing_compiler import compile_pairings
from pd_matcher.match.pipeline import _build_context
from pd_matcher.match.pipeline import match_record
from pd_matcher.match.scorers.context import ScorerContext
from pd_matcher.match.scorers.name import score_author
from pd_matcher.match.scorers.name import score_publisher
from pd_matcher.match.scorers.title import score_title
from pd_matcher.match.scorers.year import score_year
from pd_matcher.models import MarcRecord
from pd_matcher.models import NyplRenRecord
from pd_matcher.parsers.marc import MarcParseStats
from pd_matcher.parsers.marc import _build_record
from pd_matcher.parsers.marc import _RECORD_TAG
from pd_matcher.index.lookup import NyplIndexLookup

_INDEX = Path("caches/cce.lmdb")
_FIXTURE = Path("data/candidate_marc_file.marcxml")
_SAMPLE_TARGET = 3_000
_PROGRESS_LOG = Path("/tmp/agent-progress.log")


def _log(message: str) -> None:
    """Write a timestamped milestone to stderr and the shared progress log."""
    print(message, file=stderr, flush=True)
    with _PROGRESS_LOG.open("a", encoding="utf-8") as handle:
        handle.write(f"{message}\n")


def _iter_in_scope(path: Path, min_year: int, limit: int) -> Iterator[MarcRecord]:
    """Yield up to ``limit`` in-scope :class:`MarcRecord` objects from ``path``.

    Mirrors :func:`pd_matcher.parsers.marc.iter_marc_records`' streaming
    teardown, applying :func:`is_eligible` to the raw element before building
    the typed record so only moving-wall monograph BOOK records are kept.
    """
    counters = MarcParseStats()
    yielded = 0
    context = iterparse(str(path), events=("end",), tag=_RECORD_TAG)
    for _event, elem in context:
        if is_eligible(elem, min_year):
            record = _build_record(elem, counters)
            if record is not None:
                yield record
                yielded += 1
        elem.clear()
        previous = elem.getprevious()
        while previous is not None:
            del elem.getparent()[0]
            previous = elem.getprevious()
        if yielded >= limit:
            break


def _best_evidence(candidates: tuple[Evidence, ...]) -> Evidence:
    """Return the highest-scoring non-skipped Evidence, or the first if all skip."""
    best = candidates[0]
    best_score = best.score if not best.skipped else -1.0
    for evidence in candidates[1:]:
        score = evidence.score if not evidence.skipped else -1.0
        if score > best_score:
            best_score = score
            best = evidence
    return best


def _score_renewal(
    marc: MarcRecord,
    renewal: NyplRenRecord,
    ctx: ScorerContext,
    combiner: Combiner,
    calibrator: PlattCalibrator | None,
) -> float:
    """Return the calibrated weighted-mean score of one MARC↔renewal pair.

    Scores the renewal's title/author/claimants and original-registration
    year against the MARC record with the production scorers and combiner,
    keeping the best title Evidence over the MARC title fields and the best
    author Evidence over the MARC author fields (mirroring the registration
    pathway's per-group best-pairing selection). The calibrator is applied
    exactly as :func:`match_record` applies it.
    """
    title_evidence = _best_evidence(
        tuple(
            score_title(marc_value, renewal.title, ctx)
            for marc_value in (marc.title, marc.title_main)
            if marc_value
        )
        or (score_title(marc.title, renewal.title, ctx),)
    )
    author_evidence = _best_evidence(
        tuple(
            score_author(marc_value, renewal.author, ctx)
            for marc_value in (marc.main_author, marc.statement_of_responsibility)
            if marc_value
        )
        or (score_author(marc.main_author, renewal.author, ctx),)
    )
    publisher_evidence = score_publisher(marc.publisher, renewal.claimants, ctx)
    renewal_year = renewal.odat.year if renewal.odat is not None else None
    year_evidence = score_year(marc.publication_year, renewal_year, ctx)
    combined = combiner.combine(
        (title_evidence, author_evidence, publisher_evidence, year_evidence)
    )
    if calibrator is not None:
        combined = CombinedScore(
            raw=combined.raw, calibrated=calibrate(combined.raw, calibrator)
        )
    return combined.calibrated


def _best_renewal_score(
    marc: MarcRecord,
    lookup: NyplIndexLookup,
    ctx: ScorerContext,
    combiner: Combiner,
    calibrator: PlattCalibrator | None,
    window: int,
) -> float:
    """Return the best calibrated renewal score for ``marc`` (0.0 if no candidate)."""
    best = 0.0
    for renewal in lookup.candidates_for_renewal(marc, window):
        best = max(best, _score_renewal(marc, renewal, ctx, combiner, calibrator))
    return best


def main() -> None:
    """Run the gate and print the N / X / Y / Y% report."""
    min_year = default_min_year()
    matching_config = _override_matching_config(
        _load_default_matching_config(), scorer="weighted_mean"
    )
    pairings = compile_pairings(_load_default_pairing_config())
    parent = _INDEX.parent
    idf = load_or_build_idf(parent / _IDF_CACHE_NAME, lambda: NyplIndexLookup(_INDEX))
    author_idf = load_or_build_author_idf(
        parent / _AUTHOR_IDF_CACHE_NAME, lambda: NyplIndexLookup(_INDEX)
    )
    publisher_idf = load_or_build_publisher_idf(
        parent / _PUBLISHER_IDF_CACHE_NAME, lambda: NyplIndexLookup(_INDEX)
    )
    calibrator = _load_calibrator(parent)
    combiner = build_combiner(matching_config, learned_model_dir=None)
    floor = matching_config.min_combined_score / 100.0
    window = matching_config.year_window
    _log(
        f"renewal-gate: loaded index/idf/combiner; min_year={min_year} "
        f"floor={matching_config.min_combined_score} window={window} "
        f"calibrator={'yes' if calibrator is not None else 'no'}"
    )

    sample = list(_iter_in_scope(_FIXTURE, min_year, _SAMPLE_TARGET))
    _log(f"renewal-gate: sample {len(sample)} in-scope records")

    common = {
        "config": matching_config,
        "idf": idf,
        "author_idf": author_idf,
        "publisher_idf": publisher_idf,
        "calibrator": calibrator,
        "combiner": combiner,
        "pairings": pairings,
    }

    lookup = NyplIndexLookup(_INDEX)
    start = perf_counter()
    no_reg_match = 0
    renewal_only_match = 0
    for index, marc in enumerate(sample):
        result = match_record(marc, lookup=lookup, **common)
        if result.best is not None:
            continue
        no_reg_match += 1
        ctx = _build_context(marc, idf, author_idf, publisher_idf, matching_config)
        if _best_renewal_score(marc, lookup, ctx, combiner, calibrator, window) >= floor:
            renewal_only_match += 1
        if (index + 1) % 250 == 0:
            _log(
                f"renewal-gate: {index + 1}/{len(sample)} "
                f"no_reg={no_reg_match} renewal_only={renewal_only_match}"
            )
    lookup.close()

    sampled = len(sample)
    pct = (renewal_only_match / sampled * 100.0) if sampled else 0.0
    print("\n===== RENEWAL PATHWAY GATE (#45, fixture-sample proxy) =====")
    print(f"N  in-scope sampled            : {sampled}")
    print(f"X  no registration match       : {no_reg_match}")
    print(f"Y  confident renewal-only match: {renewal_only_match}")
    print(f"Y% of N (renewal ceiling)      : {pct:.2f}%")
    print(f"elapsed                        : {perf_counter() - start:.1f}s")
    print(
        "NOTE: fixture proxy — candidate_marc_file.marcxml is a raw ~18%-in-scope "
        "test fixture, not the full catalog; full-corpus gate would use "
        "data/corpus.marcxml + production pairs90.jsonl."
    )


if __name__ == "__main__":
    main()
