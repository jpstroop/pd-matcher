"""Tests for :mod:`pd_matcher.eval.ground_truth`."""

from logging import WARNING
from pathlib import Path

from pytest import LogCaptureFixture

from pd_groundtruth.label_vault import SCHEMA_VERSION
from pd_groundtruth.label_vault import MarcIdentifiers
from pd_groundtruth.label_vault import MatchSource
from pd_groundtruth.label_vault import VaultEntry
from pd_groundtruth.label_vault import upsert_entry
from pd_matcher.config.loader import load_pairing_config
from pd_matcher.config.schemas import MatchingConfig
from pd_matcher.config.schemas import PairingConfig
from pd_matcher.eval.ground_truth import EvalReport
from pd_matcher.eval.ground_truth import run_eval
from pd_matcher.index.builder import build_index

_FIXTURES = Path(__file__).resolve().parents[2] / "fixtures"
_PAIRINGS = (
    Path(__file__).resolve().parents[3]
    / "src"
    / "pd_matcher"
    / "config"
    / "defaults"
    / "field_pairings.yaml"
)

_MARCXML_PROLOG = (
    "<?xml version='1.0' encoding='UTF-8'?>\n<collection xmlns='http://www.loc.gov/MARC21/slim'>\n"
)
_MARCXML_EPILOG = "</collection>\n"


def _pairing_config() -> PairingConfig:
    """Return the shipped default field-pairing configuration."""
    return load_pairing_config(_PAIRINGS)


def _build_index(tmp_path: Path) -> Path:
    """Stand up a tiny LMDB env from the shared fixtures."""
    reg_dir = tmp_path / "reg"
    ren_dir = tmp_path / "ren"
    reg_dir.mkdir()
    ren_dir.mkdir()
    (reg_dir / "tiny_reg.xml").write_bytes((_FIXTURES / "tiny_reg.xml").read_bytes())
    (ren_dir / "tiny_ren.tsv").write_bytes((_FIXTURES / "tiny_ren.tsv").read_bytes())
    out_path = tmp_path / "idx.lmdb"
    build_index(reg_dir=reg_dir, ren_dir=ren_dir, out_path=out_path)
    return out_path


def _matching_config() -> MatchingConfig:
    """A permissive :class:`MatchingConfig` so the tiny corpus produces matches."""
    return MatchingConfig(
        title_weight=0.50,
        author_weight=0.20,
        publisher_weight=0.10,
        edition_weight=0.05,
        lccn_weight=0.10,
        isbn_weight=0.05,
        extent_weight=0.0,
        volume_weight=0.0,
        year_window=2,
        min_combined_score=1.0,
        scorer="weighted_mean",
    )


def _marc_record_xml(
    control_id: str,
    title: str,
    author: str,
    publisher: str,
    year: str,
    *,
    language: str = "eng",
) -> str:
    """Return one MARCXML <record> for the test pool."""
    return (
        "  <record>\n"
        f"    <controlfield tag='001'>{control_id}</controlfield>\n"
        f"    <controlfield tag='008'>200718s{year}    xxu           000 0 {language}  </controlfield>\n"  # noqa: E501
        f"    <datafield ind1='1' ind2=' ' tag='100'><subfield code='a'>{author}</subfield></datafield>\n"  # noqa: E501
        f"    <datafield ind1='0' ind2='0' tag='245'><subfield code='a'>{title}</subfield></datafield>\n"  # noqa: E501
        f"    <datafield ind1=' ' ind2=' ' tag='260'><subfield code='a'>New York :</subfield><subfield code='b'>{publisher},</subfield><subfield code='c'>{year}.</subfield></datafield>\n"  # noqa: E501
        "  </record>\n"
    )


def _write_pool(pool_root: Path, records: tuple[str, ...]) -> None:
    """Write ``records`` into ``<pool>/eng/shard.xml``."""
    eng = pool_root / "eng"
    eng.mkdir(parents=True)
    body = "".join(records)
    (eng / "shard.xml").write_text(
        _MARCXML_PROLOG + body + _MARCXML_EPILOG,
        encoding="utf-8",
    )


def _vault_entry(
    *,
    marc_control_id: str,
    nypl_uuid: str,
    verdict: str,
    timestamp: str = "2026-05-22T10:00:00+00:00",
    match_source: MatchSource | None = None,
) -> VaultEntry:
    """Return a minimal :class:`VaultEntry` for the synthetic vault."""
    return VaultEntry(
        schema=SCHEMA_VERSION,
        marc_control_id=marc_control_id,
        nypl_uuid=nypl_uuid,
        verdict=verdict,
        note=None,
        labeled_at=timestamp,
        labeler="test",
        marc_identifiers=MarcIdentifiers(lccn=None, oclc=None, isbns=()),
        match_source=match_source,
    )


def _seed_vault(vault_path: Path, entries: tuple[VaultEntry, ...]) -> None:
    for entry in entries:
        upsert_entry(vault_path, entry)


def _standard_marc_records() -> tuple[str, ...]:
    """Return MARCXML for one record matching UUID-0001 in the tiny index."""
    return (
        _marc_record_xml(
            control_id="marc-aaa",
            title="A study of widgets",
            author="Smith, John",
            publisher="Acme Press",
            year="1940",
        ),
    )


def _standard_vault(path: Path) -> None:
    """Seed ``path`` with a match against UUID-0001 and a no_match vs UUID-0002."""
    _seed_vault(
        path,
        (
            _vault_entry(
                marc_control_id="marc-aaa",
                nypl_uuid="UUID-0001",
                verdict="match",
            ),
            _vault_entry(
                marc_control_id="marc-aaa",
                nypl_uuid="UUID-0002",
                verdict="no_match",
            ),
        ),
    )


def test_run_eval_returns_populated_report(tmp_path: Path) -> None:
    """Vault has one match + one no_match -> aggregate counts populate correctly."""
    index_path = _build_index(tmp_path)
    pool_path = tmp_path / "pool"
    _write_pool(pool_path, _standard_marc_records())
    vault_path = tmp_path / "vault.jsonl"
    _standard_vault(vault_path)
    report = run_eval(
        vault_path=vault_path,
        pool_path=pool_path,
        index_path=index_path,
        matching_config=_matching_config(),
        pairing_config=_pairing_config(),
    )
    assert isinstance(report, EvalReport)
    assert report.pairs_evaluated == 2
    assert report.pairs_positive == 1
    assert report.pairs_negative == 1
    assert report.pairs_unsure_excluded == 0
    assert report.marcs_evaluated == 1
    assert 0.0 <= report.precision <= 1.0
    assert 0.0 <= report.recall <= 1.0
    assert 0.0 <= report.f1 <= 1.0
    assert 0.0 <= report.auc_roc <= 1.0
    assert 0.0 <= report.average_precision <= 1.0
    assert len(report.threshold_sweep) == 21
    assert report.elapsed_seconds >= 0.0


def test_run_eval_correct_top_drives_precision_and_recall(tmp_path: Path) -> None:
    """A correctly-matched top prediction gives precision=recall=1.0."""
    index_path = _build_index(tmp_path)
    pool_path = tmp_path / "pool"
    _write_pool(pool_path, _standard_marc_records())
    vault_path = tmp_path / "vault.jsonl"
    _standard_vault(vault_path)
    report = run_eval(
        vault_path=vault_path,
        pool_path=pool_path,
        index_path=index_path,
        matching_config=_matching_config(),
        pairing_config=_pairing_config(),
    )
    assert report.marcs_with_matcher_top == 1
    assert report.marcs_with_correct_top == 1
    assert report.precision == 1.0
    assert report.recall == 1.0
    assert report.f1 == 1.0


def test_run_eval_excludes_unsure_entries(tmp_path: Path) -> None:
    """``unsure`` verdicts increment the excluded count but never scored."""
    index_path = _build_index(tmp_path)
    pool_path = tmp_path / "pool"
    _write_pool(pool_path, _standard_marc_records())
    vault_path = tmp_path / "vault.jsonl"
    _seed_vault(
        vault_path,
        (
            _vault_entry(
                marc_control_id="marc-aaa",
                nypl_uuid="UUID-0001",
                verdict="match",
            ),
            _vault_entry(
                marc_control_id="marc-aaa",
                nypl_uuid="UUID-0002",
                verdict="unsure",
            ),
        ),
    )
    report = run_eval(
        vault_path=vault_path,
        pool_path=pool_path,
        index_path=index_path,
        matching_config=_matching_config(),
        pairing_config=_pairing_config(),
    )
    assert report.pairs_evaluated == 1
    assert report.pairs_positive == 1
    assert report.pairs_negative == 0
    assert report.pairs_unsure_excluded == 1


def test_run_eval_excludes_renewal_arm_pairs(
    tmp_path: Path,
    caplog: LogCaptureFixture,
) -> None:
    """``match_source='renewal'`` pairs are dropped from both passes, not scored.

    The paused renewal arm carries a renewal-record ``nypl_uuid`` the
    registration lookup cannot resolve; the eval must skip it (logging the
    count) rather than emit a ``cce_not_in_index`` miss.
    """
    from logging import INFO

    index_path = _build_index(tmp_path)
    pool_path = tmp_path / "pool"
    _write_pool(pool_path, _standard_marc_records())
    vault_path = tmp_path / "vault.jsonl"
    _seed_vault(
        vault_path,
        (
            _vault_entry(
                marc_control_id="marc-aaa",
                nypl_uuid="UUID-0001",
                verdict="match",
            ),
            _vault_entry(
                marc_control_id="marc-ren",
                nypl_uuid="REN-9999",
                verdict="match",
                match_source="renewal",
            ),
        ),
    )
    with caplog.at_level(INFO, logger="pd_matcher.eval.ground_truth"):
        report = run_eval(
            vault_path=vault_path,
            pool_path=pool_path,
            index_path=index_path,
            matching_config=_matching_config(),
            pairing_config=_pairing_config(),
        )
    assert report.pairs_evaluated == 1
    assert report.pairs_positive == 1
    assert report.marcs_evaluated == 1
    assert any("renewal_arm_excluded count=1" in r.getMessage() for r in caplog.records)
    assert not any("REN-9999" in r.getMessage() for r in caplog.records)


def test_run_eval_logs_warning_when_marc_missing_from_pool(
    tmp_path: Path,
    caplog: LogCaptureFixture,
) -> None:
    """A vault MARC missing from the pool is skipped + logged at WARNING."""
    index_path = _build_index(tmp_path)
    pool_path = tmp_path / "pool"
    _write_pool(pool_path, ())
    vault_path = tmp_path / "vault.jsonl"
    _seed_vault(
        vault_path,
        (
            _vault_entry(
                marc_control_id="marc-aaa",
                nypl_uuid="UUID-0001",
                verdict="match",
            ),
        ),
    )
    with caplog.at_level(WARNING, logger="pd_matcher.eval.ground_truth"):
        report = run_eval(
            vault_path=vault_path,
            pool_path=pool_path,
            index_path=index_path,
            matching_config=_matching_config(),
            pairing_config=_pairing_config(),
        )
    assert report.pairs_evaluated == 0
    assert any("marc_not_in_pool" in record.message for record in caplog.records)


def test_run_eval_logs_warning_when_cce_missing_from_index(
    tmp_path: Path,
    caplog: LogCaptureFixture,
) -> None:
    """A vault CCE UUID missing from the index is skipped + logged at WARNING."""
    index_path = _build_index(tmp_path)
    pool_path = tmp_path / "pool"
    _write_pool(pool_path, _standard_marc_records())
    vault_path = tmp_path / "vault.jsonl"
    _seed_vault(
        vault_path,
        (
            _vault_entry(
                marc_control_id="marc-aaa",
                nypl_uuid="UUID-NOPE",
                verdict="no_match",
            ),
        ),
    )
    with caplog.at_level(WARNING, logger="pd_matcher.eval.ground_truth"):
        report = run_eval(
            vault_path=vault_path,
            pool_path=pool_path,
            index_path=index_path,
            matching_config=_matching_config(),
            pairing_config=_pairing_config(),
        )
    assert report.pairs_evaluated == 0
    assert any("cce_not_in_index" in record.message for record in caplog.records)


def test_run_eval_logs_warning_when_gt_marc_missing(
    tmp_path: Path,
    caplog: LogCaptureFixture,
) -> None:
    """A ground-truth MARC missing from the pool is logged in Pass B."""
    index_path = _build_index(tmp_path)
    pool_path = tmp_path / "pool"
    _write_pool(pool_path, ())
    vault_path = tmp_path / "vault.jsonl"
    _seed_vault(
        vault_path,
        (
            _vault_entry(
                marc_control_id="marc-missing",
                nypl_uuid="UUID-0001",
                verdict="match",
            ),
        ),
    )
    with caplog.at_level(WARNING, logger="pd_matcher.eval.ground_truth"):
        report = run_eval(
            vault_path=vault_path,
            pool_path=pool_path,
            index_path=index_path,
            matching_config=_matching_config(),
            pairing_config=_pairing_config(),
        )
    assert report.marcs_evaluated == 0
    assert any("gt_marc_not_in_pool" in record.message for record in caplog.records)


def test_run_eval_handles_empty_vault(tmp_path: Path) -> None:
    """An empty vault yields zero metrics, not a divide-by-zero."""
    index_path = _build_index(tmp_path)
    pool_path = tmp_path / "pool"
    _write_pool(pool_path, ())
    vault_path = tmp_path / "vault.jsonl"
    vault_path.write_text("", encoding="utf-8")
    report = run_eval(
        vault_path=vault_path,
        pool_path=pool_path,
        index_path=index_path,
        matching_config=_matching_config(),
        pairing_config=_pairing_config(),
    )
    assert report.pairs_evaluated == 0
    assert report.precision == 0.0
    assert report.recall == 0.0
    assert report.f1 == 0.0
    assert report.auc_roc == 0.0
    assert report.average_precision == 0.0
    assert report.threshold_sweep == ()


def test_run_eval_year_window_zero_blocks_year_drift(tmp_path: Path) -> None:
    """``year_window=0`` blocks a MARC labeled at a drifted year from matching."""
    index_path = _build_index(tmp_path)
    pool_path = tmp_path / "pool"
    _write_pool(
        pool_path,
        (
            _marc_record_xml(
                control_id="marc-drift",
                title="A study of widgets",
                author="Smith, John",
                publisher="Acme Press",
                year="1945",
            ),
        ),
    )
    vault_path = tmp_path / "vault.jsonl"
    _seed_vault(
        vault_path,
        (
            _vault_entry(
                marc_control_id="marc-drift",
                nypl_uuid="UUID-0001",
                verdict="match",
            ),
        ),
    )
    narrow = MatchingConfig(
        title_weight=0.50,
        author_weight=0.20,
        publisher_weight=0.10,
        edition_weight=0.05,
        lccn_weight=0.10,
        isbn_weight=0.05,
        extent_weight=0.0,
        volume_weight=0.0,
        year_window=0,
        min_combined_score=1.0,
        scorer="weighted_mean",
    )
    report = run_eval(
        vault_path=vault_path,
        pool_path=pool_path,
        index_path=index_path,
        matching_config=narrow,
        pairing_config=_pairing_config(),
    )
    assert report.marcs_with_matcher_top == 0
    assert report.marcs_with_correct_top == 0
    assert report.precision == 0.0


def test_run_eval_year_window_five_admits_drifted_match(tmp_path: Path) -> None:
    """``year_window=5`` admits the same drifted candidate as a match."""
    index_path = _build_index(tmp_path)
    pool_path = tmp_path / "pool"
    _write_pool(
        pool_path,
        (
            _marc_record_xml(
                control_id="marc-drift",
                title="A study of widgets",
                author="Smith, John",
                publisher="Acme Press",
                year="1945",
            ),
        ),
    )
    vault_path = tmp_path / "vault.jsonl"
    _seed_vault(
        vault_path,
        (
            _vault_entry(
                marc_control_id="marc-drift",
                nypl_uuid="UUID-0001",
                verdict="match",
            ),
        ),
    )
    wide = MatchingConfig(
        title_weight=0.50,
        author_weight=0.20,
        publisher_weight=0.10,
        edition_weight=0.05,
        lccn_weight=0.10,
        isbn_weight=0.05,
        extent_weight=0.0,
        volume_weight=0.0,
        year_window=5,
        min_combined_score=1.0,
        scorer="weighted_mean",
    )
    report = run_eval(
        vault_path=vault_path,
        pool_path=pool_path,
        index_path=index_path,
        matching_config=wide,
        pairing_config=_pairing_config(),
    )
    assert report.marcs_with_matcher_top == 1
    assert report.marcs_with_correct_top == 1


def test_run_eval_top_disagrees_with_ground_truth(tmp_path: Path) -> None:
    """A matched top whose UUID differs from the vault GT increments with_top only."""
    index_path = _build_index(tmp_path)
    pool_path = tmp_path / "pool"
    _write_pool(pool_path, _standard_marc_records())
    vault_path = tmp_path / "vault.jsonl"
    _seed_vault(
        vault_path,
        (
            _vault_entry(
                marc_control_id="marc-aaa",
                nypl_uuid="UUID-0002",
                verdict="match",
            ),
        ),
    )
    report = run_eval(
        vault_path=vault_path,
        pool_path=pool_path,
        index_path=index_path,
        matching_config=_matching_config(),
        pairing_config=_pairing_config(),
    )
    assert report.marcs_evaluated == 1
    assert report.marcs_with_matcher_top == 1
    assert report.marcs_with_correct_top == 0
    assert report.precision == 0.0


def test_run_eval_latest_verdict_wins(tmp_path: Path) -> None:
    """A re-labeled pair: only the latest verdict shows up in the counts."""
    index_path = _build_index(tmp_path)
    pool_path = tmp_path / "pool"
    _write_pool(pool_path, _standard_marc_records())
    vault_path = tmp_path / "vault.jsonl"
    _seed_vault(
        vault_path,
        (
            _vault_entry(
                marc_control_id="marc-aaa",
                nypl_uuid="UUID-0001",
                verdict="match",
                timestamp="2026-05-01T10:00:00+00:00",
            ),
            _vault_entry(
                marc_control_id="marc-aaa",
                nypl_uuid="UUID-0001",
                verdict="no_match",
                timestamp="2026-05-22T10:00:00+00:00",
            ),
        ),
    )
    report = run_eval(
        vault_path=vault_path,
        pool_path=pool_path,
        index_path=index_path,
        matching_config=_matching_config(),
        pairing_config=_pairing_config(),
    )
    assert report.pairs_positive == 0
    assert report.pairs_negative == 1
    assert report.marcs_evaluated == 0


def _learned_matching_config() -> MatchingConfig:
    """The permissive config with the learned scorer selected."""
    base = _matching_config()
    return MatchingConfig(
        title_weight=base.title_weight,
        author_weight=base.author_weight,
        publisher_weight=base.publisher_weight,
        edition_weight=base.edition_weight,
        lccn_weight=base.lccn_weight,
        isbn_weight=base.isbn_weight,
        extent_weight=base.extent_weight,
        volume_weight=base.volume_weight,
        year_window=base.year_window,
        min_combined_score=base.min_combined_score,
        scorer="learned",
    )


def _write_tiny_learned_model(directory: Path) -> None:
    """Train and persist a tiny learned-model artifact under ``directory``."""
    from lightgbm import LGBMClassifier
    from numpy import asarray
    from numpy import float64
    from numpy import int64

    from pd_matcher.match.combiners.features import feature_names
    from pd_matcher.match.combiners.learned import model_metadata
    from pd_matcher.match.combiners.learned import save_learned_model

    n_features = len(feature_names())
    rows = [[0.9 if i % 2 == 0 else 0.1] + [0.0] * (n_features - 1) for i in range(20)]
    labels = [1 if i % 2 == 0 else 0 for i in range(20)]
    model = LGBMClassifier(
        max_depth=3,
        num_leaves=8,
        min_data_in_leaf=5,
        n_estimators=10,
        random_state=0,
        verbose=-1,
        n_jobs=1,
    )
    model.fit(asarray(rows, dtype=float64), asarray(labels, dtype=int64).astype(float64))
    meta = model_metadata(
        model.booster_,
        n_positive=10,
        n_negative=10,
        max_depth=3,
        num_leaves=8,
        min_data_in_leaf=5,
        lambda_l2=1.0,
        n_estimators=10,
        class_weight="balanced",
    )
    save_learned_model(model.booster_, meta, directory)


def test_run_eval_threads_learned_model_through_both_passes(tmp_path: Path) -> None:
    """run_eval scores via the learned combiner when given a model directory."""
    index_path = _build_index(tmp_path)
    pool_path = tmp_path / "pool"
    _write_pool(pool_path, _standard_marc_records())
    vault_path = tmp_path / "vault.jsonl"
    _standard_vault(vault_path)
    model_dir = tmp_path / "model"
    _write_tiny_learned_model(model_dir)
    report = run_eval(
        vault_path=vault_path,
        pool_path=pool_path,
        index_path=index_path,
        matching_config=_learned_matching_config(),
        pairing_config=_pairing_config(),
        learned_model_dir=model_dir,
    )
    assert isinstance(report, EvalReport)
    assert report.pairs_evaluated == 2
