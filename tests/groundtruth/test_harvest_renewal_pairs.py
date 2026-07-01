"""Unit tests for the verified MARC↔renewal training-pair harvester."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import date
from pathlib import Path
from typing import cast

from msgspec.json import decode as json_decode
from pytest import MonkeyPatch
from pytest import raises

from pd_groundtruth import harvest_renewal_pairs as module
from pd_groundtruth.build_renewal_queue import RenewalScore
from pd_groundtruth.build_renewal_queue import RenewalScoreFn
from pd_groundtruth.harvest_renewal_pairs import PROVENANCE_HARD_NEGATIVE
from pd_groundtruth.harvest_renewal_pairs import PROVENANCE_POSITIVE
from pd_groundtruth.harvest_renewal_pairs import HarvestedPair
from pd_groundtruth.harvest_renewal_pairs import harvest_renewal_pairs
from pd_groundtruth.harvest_renewal_pairs import run_harvest
from pd_groundtruth.harvest_renewal_pairs import write_harvest
from pd_groundtruth.label_vault import MarcIdentifiers
from pd_groundtruth.label_vault import MatchSource
from pd_groundtruth.label_vault import VaultEntry
from pd_matcher.cli import _load_default_matching_config
from pd_matcher.config.schemas import MatchingConfig
from pd_matcher.index.lookup import NyplIndexLookup
from pd_matcher.models import IndexedNyplRegRecord
from pd_matcher.models import MarcRecord
from pd_matcher.models import NyplRenRecord

_FIXTURE_VAULT = Path(__file__).parent.parent / "fixtures" / "harvest_vault.jsonl"


# --------------------------------------------------------------------------- #
# constructors
# --------------------------------------------------------------------------- #


def _marc(control_id: str = "marc-1", *, year: int | None = 1953) -> MarcRecord:
    return MarcRecord(
        control_id=control_id,
        title="A True Title",
        title_main="A True Title",
        main_author="An Author",
        statement_of_responsibility="by An Author",
        publisher="A Publisher",
        publication_year=year,
        language_code="eng",
    )


def _reg(
    uuid: str = "reg-1",
    *,
    was_renewed: bool = True,
    renewal_id: str | None = "R1",
) -> IndexedNyplRegRecord:
    return IndexedNyplRegRecord(
        uuid=uuid,
        title="Reg Title",
        was_renewed=was_renewed,
        renewal_id=renewal_id,
        reg_year=1953,
    )


def _renewal(
    ren_id: str = "R1", *, entry_id: str = "e1", title: str = "Renewal Title"
) -> NyplRenRecord:
    return NyplRenRecord(
        id=ren_id,
        entry_id=entry_id,
        oreg="A111111",
        odat=date(1953, 1, 1),
        rdat=date(1981, 4, 1),
        author="Renewal Author",
        title=title,
        claimants="Renewal Claimant",
        new_matter="added chapters",
    )


def _entry(
    marc: str,
    uuid: str,
    verdict: str = "match",
    match_source: MatchSource | None = "registration",
) -> VaultEntry:
    return VaultEntry(
        schema=7,
        marc_control_id=marc,
        nypl_uuid=uuid,
        verdict=verdict,
        note=None,
        labeled_at="2026-06-01T00:00:00+00:00",
        labeler="jpstroop",
        marc_identifiers=MarcIdentifiers(lccn=None, oclc=None, isbns=()),
        match_source=match_source,
    )


def _score(mapping: dict[str, float]) -> RenewalScoreFn:
    def score_fn(_marc: MarcRecord, renewal: NyplRenRecord) -> RenewalScore:
        return RenewalScore(calibrated=mapping.get(renewal.id, 0.0), evidence={"title": 0.5})

    return score_fn


def _harvest(
    entries: list[VaultEntry],
    *,
    marcs: dict[str, MarcRecord],
    regs: dict[str, IndexedNyplRegRecord],
    renewals: dict[str, NyplRenRecord],
    candidates: dict[str, tuple[NyplRenRecord, ...]] | None = None,
    scores: dict[str, float] | None = None,
    negatives_per_positive: int = 1,
) -> tuple[list[HarvestedPair], module.HarvestSummary]:
    candidates = candidates or {}
    return harvest_renewal_pairs(
        entries=entries,
        marc_lookup=marcs.get,
        reg_lookup=regs.get,
        renewal_lookup=renewals.get,
        renewal_candidates=lambda marc, _window: candidates.get(marc.control_id, ()),
        score_fn=_score(scores or {}),
        window=0,
        negatives_per_positive=negatives_per_positive,
    )


# --------------------------------------------------------------------------- #
# _is_registration_pathway_match
# --------------------------------------------------------------------------- #


def test_registration_pathway_match_accepts_registration_none_and_both() -> None:
    assert module._is_registration_pathway_match(_entry("m", "u", match_source="registration"))
    assert module._is_registration_pathway_match(_entry("m", "u", match_source=None))
    assert module._is_registration_pathway_match(_entry("m", "u", match_source="both"))


def test_registration_pathway_match_rejects_renewal_and_non_match() -> None:
    assert not module._is_registration_pathway_match(_entry("m", "u", match_source="renewal"))
    assert not module._is_registration_pathway_match(_entry("m", "u", verdict="no_match"))
    assert not module._is_registration_pathway_match(_entry("m", "u", verdict="unsure"))


# --------------------------------------------------------------------------- #
# _marc_author
# --------------------------------------------------------------------------- #


def test_marc_author_prefers_main_author() -> None:
    assert module._marc_author(_marc()) == "An Author"


def test_marc_author_falls_back_to_statement_of_responsibility() -> None:
    marc = MarcRecord(
        control_id="m",
        title="t",
        title_main="t",
        main_author=None,
        statement_of_responsibility="by Someone",
    )
    assert module._marc_author(marc) == "by Someone"


# --------------------------------------------------------------------------- #
# harvest_renewal_pairs — positives
# --------------------------------------------------------------------------- #


def test_joined_registration_emits_positive() -> None:
    pairs, summary = _harvest(
        [_entry("marc-1", "reg-1")],
        marcs={"marc-1": _marc("marc-1")},
        regs={"reg-1": _reg("reg-1", renewal_id="R1")},
        renewals={"R1": _renewal("R1")},
        scores={"R1": 0.92},
    )
    assert summary.vault_matches_examined == 1
    assert summary.joined == 1
    assert summary.positives == 1
    assert summary.negatives == 0
    assert len(pairs) == 1
    positive = pairs[0]
    assert positive.label == "match"
    assert positive.provenance == PROVENANCE_POSITIVE
    assert positive.marc_control_id == "marc-1"
    assert positive.marc_title == "A True Title"
    assert positive.marc_author == "An Author"
    assert positive.marc_publisher == "A Publisher"
    assert positive.marc_year == 1953
    assert positive.renewal_id == "R1"
    assert positive.renewal_title == "Renewal Title"
    assert positive.renewal_claimants == "Renewal Claimant"
    assert positive.renewal_oreg == "A111111"
    assert positive.renewal_odat == "1953-01-01"
    assert positive.score == 0.92


def test_legacy_none_match_source_is_registration_pathway_positive() -> None:
    _pairs, summary = _harvest(
        [_entry("marc-1", "reg-1", match_source=None)],
        marcs={"marc-1": _marc("marc-1")},
        regs={"reg-1": _reg("reg-1")},
        renewals={"R1": _renewal("R1")},
    )
    assert summary.positives == 1


def test_renewal_without_odat_emits_null_odat() -> None:
    renewal = NyplRenRecord(id="R1", entry_id="e", title="T", claimants="C")
    pairs, _summary = _harvest(
        [_entry("marc-1", "reg-1")],
        marcs={"marc-1": _marc("marc-1")},
        regs={"reg-1": _reg("reg-1")},
        renewals={"R1": renewal},
    )
    assert pairs[0].renewal_odat is None


# --------------------------------------------------------------------------- #
# harvest_renewal_pairs — skips
# --------------------------------------------------------------------------- #


def test_unjoined_registration_is_skipped() -> None:
    pairs, summary = _harvest(
        [_entry("marc-1", "reg-1")],
        marcs={"marc-1": _marc("marc-1")},
        regs={"reg-1": _reg("reg-1", was_renewed=False, renewal_id=None)},
        renewals={},
    )
    assert pairs == []
    assert summary.registration_not_joined == 1
    assert summary.positives == 0


def test_renewed_flag_without_renewal_id_is_not_joined() -> None:
    _pairs, summary = _harvest(
        [_entry("marc-1", "reg-1")],
        marcs={"marc-1": _marc("marc-1")},
        regs={"reg-1": _reg("reg-1", was_renewed=True, renewal_id=None)},
        renewals={},
    )
    assert summary.registration_not_joined == 1


def test_non_match_and_renewal_pathway_entries_are_not_examined() -> None:
    _pairs, summary = _harvest(
        [
            _entry("marc-1", "reg-1", verdict="no_match"),
            _entry("marc-2", "reg-2", verdict="unsure"),
            _entry("marc-3", "reg-3", match_source="renewal"),
        ],
        marcs={"marc-1": _marc("marc-1"), "marc-2": _marc("marc-2"), "marc-3": _marc("marc-3")},
        regs={"reg-1": _reg("reg-1"), "reg-2": _reg("reg-2"), "reg-3": _reg("reg-3")},
        renewals={"R1": _renewal("R1")},
    )
    assert summary.vault_matches_examined == 0
    assert summary.positives == 0


def test_missing_marc_is_counted_and_skipped() -> None:
    _pairs, summary = _harvest(
        [_entry("marc-gone", "reg-1")],
        marcs={},
        regs={"reg-1": _reg("reg-1")},
        renewals={"R1": _renewal("R1")},
    )
    assert summary.missing_marc == 1
    assert summary.positives == 0


def test_missing_registration_is_counted_and_skipped() -> None:
    _pairs, summary = _harvest(
        [_entry("marc-1", "reg-gone")],
        marcs={"marc-1": _marc("marc-1")},
        regs={},
        renewals={"R1": _renewal("R1")},
    )
    assert summary.missing_registration == 1
    assert summary.positives == 0


def test_missing_joined_renewal_is_counted_and_skipped() -> None:
    _pairs, summary = _harvest(
        [_entry("marc-1", "reg-1")],
        marcs={"marc-1": _marc("marc-1")},
        regs={"reg-1": _reg("reg-1", renewal_id="R-gone")},
        renewals={},
    )
    assert summary.renewal_missing == 1
    assert summary.positives == 0


# --------------------------------------------------------------------------- #
# harvest_renewal_pairs — hard negatives
# --------------------------------------------------------------------------- #


def test_hard_negative_excludes_true_renewal_and_picks_top_wrong_candidate() -> None:
    true_renewal = _renewal("R1", entry_id="e-true")
    look_alike_hi = _renewal("R-hi", entry_id="e-hi", title="Look-alike High")
    look_alike_lo = _renewal("R-lo", entry_id="e-lo", title="Look-alike Low")
    pairs, summary = _harvest(
        [_entry("marc-1", "reg-1")],
        marcs={"marc-1": _marc("marc-1")},
        regs={"reg-1": _reg("reg-1", renewal_id="R1")},
        renewals={"R1": true_renewal},
        candidates={"marc-1": (look_alike_lo, true_renewal, look_alike_hi)},
        scores={"R1": 0.99, "R-hi": 0.8, "R-lo": 0.2},
        negatives_per_positive=1,
    )
    assert summary.positives == 1
    assert summary.negatives == 1
    negatives = [p for p in pairs if p.provenance == PROVENANCE_HARD_NEGATIVE]
    assert len(negatives) == 1
    negative = negatives[0]
    assert negative.label == "no_match"
    assert negative.renewal_id == "R-hi"
    assert negative.score == 0.8


def test_hard_negative_count_is_configurable() -> None:
    true_renewal = _renewal("R1")
    pairs, summary = _harvest(
        [_entry("marc-1", "reg-1")],
        marcs={"marc-1": _marc("marc-1")},
        regs={"reg-1": _reg("reg-1", renewal_id="R1")},
        renewals={"R1": true_renewal},
        candidates={
            "marc-1": (_renewal("R-a", entry_id="a"), _renewal("R-b", entry_id="b"), true_renewal)
        },
        scores={"R-a": 0.7, "R-b": 0.6},
        negatives_per_positive=2,
    )
    assert summary.negatives == 2
    ids = {p.renewal_id for p in pairs if p.provenance == PROVENANCE_HARD_NEGATIVE}
    assert ids == {"R-a", "R-b"}


def test_zero_negatives_emits_only_positive() -> None:
    pairs, summary = _harvest(
        [_entry("marc-1", "reg-1")],
        marcs={"marc-1": _marc("marc-1")},
        regs={"reg-1": _reg("reg-1", renewal_id="R1")},
        renewals={"R1": _renewal("R1")},
        candidates={"marc-1": (_renewal("R-x", entry_id="x"),)},
        scores={"R-x": 0.9},
        negatives_per_positive=0,
    )
    assert summary.negatives == 0
    assert len(pairs) == 1


def test_rank_hard_negatives_returns_empty_for_non_positive_limit() -> None:
    assert module._rank_hard_negatives(_marc(), (_renewal("R-x"),), _score({}), "R1", 0) == []


def test_rank_hard_negatives_orders_descending() -> None:
    ranked = module._rank_hard_negatives(
        _marc(),
        (_renewal("R-lo", entry_id="lo"), _renewal("R-hi", entry_id="hi")),
        _score({"R-lo": 0.1, "R-hi": 0.9}),
        "R1",
        5,
    )
    assert [renewal.id for renewal, _score in ranked] == ["R-hi", "R-lo"]


# --------------------------------------------------------------------------- #
# write_harvest
# --------------------------------------------------------------------------- #


def test_write_harvest_round_trips_jsonl(tmp_path: Path) -> None:
    pairs, _summary = _harvest(
        [_entry("marc-1", "reg-1")],
        marcs={"marc-1": _marc("marc-1")},
        regs={"reg-1": _reg("reg-1")},
        renewals={"R1": _renewal("R1")},
    )
    out = tmp_path / "nested" / "harvest.jsonl"
    write_harvest(out, pairs)
    lines = out.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    decoded = json_decode(lines[0], type=HarvestedPair)
    assert decoded.provenance == PROVENANCE_POSITIVE
    assert decoded.renewal_id == "R1"


# --------------------------------------------------------------------------- #
# _make_renewal_score_fn
# --------------------------------------------------------------------------- #


def test_make_renewal_score_fn_wires_idf_calibrator_and_combiner(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    """The scorer builder loads the IDF caches, calibrator, and combiner once."""
    sentinel = _score({"R1": 0.5})
    seen: dict[str, object] = {}
    config = _config()
    monkeypatch.setattr(module, "load_or_build_idf", lambda *_a, **_k: "idf")
    monkeypatch.setattr(module, "load_or_build_author_idf", lambda *_a, **_k: "author")
    monkeypatch.setattr(module, "load_or_build_publisher_idf", lambda *_a, **_k: "publisher")
    monkeypatch.setattr(module, "_load_calibrator", lambda parent: "calibrator")
    monkeypatch.setattr(module, "build_combiner", lambda _c, **_k: "combiner")

    def fake_make_score_fn(*args: object) -> RenewalScoreFn:
        seen["args"] = args
        return sentinel

    monkeypatch.setattr(module, "_make_score_fn", fake_make_score_fn)
    result = module._make_renewal_score_fn(tmp_path / "cce.lmdb", config)
    assert result is sentinel
    assert seen["args"] == ("idf", "author", "publisher", config, "combiner", "calibrator")


# --------------------------------------------------------------------------- #
# run_harvest (real vault fixture + faked index wiring)
# --------------------------------------------------------------------------- #


class _FakeLookup:
    """Stand-in for ``NyplIndexLookup`` serving canned reg/renewal data.

    ``renewals`` are keyed by ``entry_id`` (as the real ``ren_by_id`` store is);
    :meth:`iter_renewals` yields them so the ``id -> entry_id`` bridge can be
    built exactly as production does.
    """

    def __init__(
        self,
        regs: dict[str, IndexedNyplRegRecord],
        renewals: dict[str, NyplRenRecord],
        candidates: dict[str, tuple[NyplRenRecord, ...]],
    ) -> None:
        self._regs = regs
        self._renewals = renewals
        self._candidates = candidates

    def __enter__(self) -> _FakeLookup:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def get_registration(self, uuid: str) -> IndexedNyplRegRecord | None:
        return self._regs.get(uuid)

    def get_renewal(self, entry_id: str) -> NyplRenRecord | None:
        return self._renewals.get(entry_id)

    def iter_renewals(self) -> Iterator[NyplRenRecord]:
        return iter(self._renewals.values())

    def candidates_for_renewal(self, marc: MarcRecord, _window: int) -> Iterator[NyplRenRecord]:
        return iter(self._candidates.get(marc.control_id, ()))


def test_make_renewal_lookup_bridges_id_to_entry_id() -> None:
    """A registration's ``renewal_id`` (the renewal ``.id``) resolves the record."""
    true_renewal = _renewal("R1", entry_id="e-true")
    lookup = _FakeLookup(regs={}, renewals={"e-true": true_renewal}, candidates={})
    renewal_lookup = module._make_renewal_lookup(cast(NyplIndexLookup, lookup))
    assert renewal_lookup("R1") is true_renewal
    assert renewal_lookup("R-unknown") is None


def _config() -> MatchingConfig:
    return _load_default_matching_config()


def _patch_run_wiring(monkeypatch: MonkeyPatch, lookup: _FakeLookup) -> None:
    monkeypatch.setattr(
        module,
        "build_marc_index_from_collection",
        lambda _path, wanted: {marc_id: _marc(marc_id) for marc_id in wanted},
    )
    monkeypatch.setattr(module, "_make_renewal_score_fn", lambda *_a, **_k: _score({"R-hi": 0.8}))
    monkeypatch.setattr(module, "NyplIndexLookup", lambda _path: lookup)


def test_run_harvest_reads_fixture_and_writes(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    """The fixture vault yields two positives (registration + legacy None)."""
    true_renewal = _renewal("R1", entry_id="e-true")
    look_alike = _renewal("R-hi", entry_id="e-hi", title="Look-alike")
    lookup = _FakeLookup(
        regs={
            "reg-joined": _reg("reg-joined", renewal_id="R1"),
            "reg-legacy-joined": _reg("reg-legacy-joined", renewal_id="R1"),
            "reg-notjoined": _reg("reg-notjoined", was_renewed=False, renewal_id=None),
        },
        renewals={"e-true": true_renewal},
        candidates={
            "marc-joined": (look_alike, true_renewal),
            "marc-legacy": (look_alike,),
        },
    )
    _patch_run_wiring(monkeypatch, lookup)
    out = tmp_path / "harvest.jsonl"
    pairs, summary = run_harvest(
        vault_path=_FIXTURE_VAULT,
        index_path=tmp_path / "cce.lmdb",
        out_path=out,
        matching_config=_config(),
        negatives_per_positive=1,
        marc_collection_path=tmp_path / "marc.xml",
    )
    assert summary.vault_matches_examined == 3
    assert summary.positives == 2
    assert summary.registration_not_joined == 1
    assert summary.negatives == 2
    written = out.read_text(encoding="utf-8").splitlines()
    assert len(written) == len(pairs) == 4
    provenances = {json_decode(line, type=HarvestedPair).provenance for line in written}
    assert provenances == {PROVENANCE_POSITIVE, PROVENANCE_HARD_NEGATIVE}


def test_run_harvest_requires_exactly_one_marc_source(tmp_path: Path) -> None:
    with raises(ValueError, match="exactly one"):
        run_harvest(
            vault_path=_FIXTURE_VAULT,
            index_path=tmp_path / "cce.lmdb",
            out_path=tmp_path / "out.jsonl",
            matching_config=_config(),
            negatives_per_positive=1,
            pool_path=tmp_path / "pool",
            marc_collection_path=tmp_path / "marc.xml",
        )
    with raises(ValueError, match="exactly one"):
        run_harvest(
            vault_path=_FIXTURE_VAULT,
            index_path=tmp_path / "cce.lmdb",
            out_path=tmp_path / "out.jsonl",
            matching_config=_config(),
            negatives_per_positive=1,
        )


def test_run_harvest_from_pool_uses_pool_index(tmp_path: Path, monkeypatch: MonkeyPatch) -> None:
    """The ``--pool`` branch resolves MARCs from the sharded pool builder."""
    calls: dict[str, object] = {}

    def fake_pool_index(pool: Path, wanted: set[str]) -> dict[str, MarcRecord]:
        calls["pool"] = pool
        return {marc_id: _marc(marc_id) for marc_id in wanted}

    monkeypatch.setattr(module, "build_marc_index", fake_pool_index)
    monkeypatch.setattr(module, "_make_renewal_score_fn", lambda *_a, **_k: _score({}))
    lookup = _FakeLookup(
        regs={
            "reg-joined": _reg("reg-joined", renewal_id="R1"),
            "reg-legacy-joined": _reg("reg-legacy-joined", renewal_id="R1"),
            "reg-notjoined": _reg("reg-notjoined", was_renewed=False, renewal_id=None),
        },
        renewals={"e1": _renewal("R1")},
        candidates={},
    )
    monkeypatch.setattr(module, "NyplIndexLookup", lambda _path: lookup)
    _pairs, summary = run_harvest(
        vault_path=_FIXTURE_VAULT,
        index_path=tmp_path / "cce.lmdb",
        out_path=tmp_path / "out.jsonl",
        matching_config=_config(),
        negatives_per_positive=0,
        pool_path=tmp_path / "pool",
    )
    assert calls["pool"] == tmp_path / "pool"
    assert summary.positives == 2
