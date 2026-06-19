"""Tests for :mod:`pd_matcher.match.combiners.train`."""

from logging import WARNING
from pathlib import Path

from pytest import LogCaptureFixture
from pytest import raises

from pd_groundtruth.label_vault import SCHEMA_VERSION
from pd_groundtruth.label_vault import MarcIdentifiers
from pd_groundtruth.label_vault import VaultEntry
from pd_groundtruth.label_vault import upsert_entry
from pd_matcher.config.loader import load_pairing_config
from pd_matcher.config.schemas import MatchingConfig
from pd_matcher.config.schemas import PairingConfig
from pd_matcher.index.builder import build_index
from pd_matcher.match.combiners.features import feature_names
from pd_matcher.match.combiners.train import build_training_matrix
from pd_matcher.match.combiners.train import train_learned_model

_FIXTURES = Path(__file__).resolve().parents[3] / "fixtures"
_PAIRINGS = (
    Path(__file__).resolve().parents[4]
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
    return load_pairing_config(_PAIRINGS)


def _matching_config() -> MatchingConfig:
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
        scorer="learned",
    )


def _build_index(tmp_path: Path) -> Path:
    reg_dir = tmp_path / "reg"
    ren_dir = tmp_path / "ren"
    reg_dir.mkdir()
    ren_dir.mkdir()
    (reg_dir / "tiny_reg.xml").write_bytes((_FIXTURES / "tiny_reg.xml").read_bytes())
    (ren_dir / "tiny_ren.tsv").write_bytes((_FIXTURES / "tiny_ren.tsv").read_bytes())
    out_path = tmp_path / "idx.lmdb"
    build_index(reg_dir=reg_dir, ren_dir=ren_dir, out_path=out_path)
    return out_path


def _marc_xml(control_id: str) -> str:
    return (
        "  <record>\n"
        f"    <controlfield tag='001'>{control_id}</controlfield>\n"
        "    <controlfield tag='008'>200718s1940    xxu           000 0 eng  </controlfield>\n"
        "    <datafield ind1='1' ind2=' ' tag='100'><subfield code='a'>Smith, John</subfield></datafield>\n"  # noqa: E501
        "    <datafield ind1='0' ind2='0' tag='245'><subfield code='a'>A study of widgets</subfield></datafield>\n"  # noqa: E501
        "    <datafield ind1=' ' ind2=' ' tag='260'><subfield code='a'>New York :</subfield><subfield code='b'>Acme Press,</subfield><subfield code='c'>1940.</subfield></datafield>\n"  # noqa: E501
        "  </record>\n"
    )


def _write_pool(pool_root: Path, control_id: str) -> None:
    eng = pool_root / "eng"
    eng.mkdir(parents=True)
    (eng / "shard.xml").write_text(
        _MARCXML_PROLOG + _marc_xml(control_id) + _MARCXML_EPILOG,
        encoding="utf-8",
    )


def _entry(marc_control_id: str, nypl_uuid: str, verdict: str, minute: int) -> VaultEntry:
    return VaultEntry(
        schema=SCHEMA_VERSION,
        marc_control_id=marc_control_id,
        nypl_uuid=nypl_uuid,
        verdict=verdict,
        note=None,
        labeled_at=f"2026-05-22T10:{minute:02d}:00+00:00",
        labeler="test",
        marc_identifiers=MarcIdentifiers(lccn=None, oclc=None, isbns=()),
    )


def _seed(vault_path: Path, entries: tuple[VaultEntry, ...]) -> None:
    for entry in entries:
        upsert_entry(vault_path, entry)


def test_build_training_matrix_has_canonical_width(tmp_path: Path) -> None:
    """The matrix width equals the canonical feature count."""
    index_path = _build_index(tmp_path)
    pool_path = tmp_path / "pool"
    _write_pool(pool_path, "marc-aaa")
    vault_path = tmp_path / "vault.jsonl"
    _seed(
        vault_path,
        (
            _entry("marc-aaa", "UUID-0001", "match", 0),
            _entry("marc-aaa", "UUID-0002", "no_match", 1),
        ),
    )
    matrix = build_training_matrix(
        vault_path=vault_path,
        pool_path=pool_path,
        index_path=index_path,
        matching_config=_matching_config(),
        pairing_config=_pairing_config(),
    )
    assert matrix.x.shape[1] == len(feature_names())
    assert matrix.n_positive == 1
    assert matrix.n_negative == 1


def test_build_training_matrix_excludes_unsure(tmp_path: Path) -> None:
    """``unsure`` verdicts never enter the training matrix."""
    index_path = _build_index(tmp_path)
    pool_path = tmp_path / "pool"
    _write_pool(pool_path, "marc-aaa")
    vault_path = tmp_path / "vault.jsonl"
    _seed(
        vault_path,
        (
            _entry("marc-aaa", "UUID-0001", "match", 0),
            _entry("marc-aaa", "UUID-0002", "unsure", 1),
        ),
    )
    matrix = build_training_matrix(
        vault_path=vault_path,
        pool_path=pool_path,
        index_path=index_path,
        matching_config=_matching_config(),
        pairing_config=_pairing_config(),
    )
    assert matrix.n_positive + matrix.n_negative == 1


def test_build_training_matrix_warns_marc_not_in_pool(
    tmp_path: Path,
    caplog: LogCaptureFixture,
) -> None:
    """A vault MARC absent from the pool is logged and skipped."""
    index_path = _build_index(tmp_path)
    pool_path = tmp_path / "pool"
    _write_pool(pool_path, "marc-aaa")
    vault_path = tmp_path / "vault.jsonl"
    _seed(vault_path, (_entry("marc-missing", "UUID-0001", "match", 0),))
    with caplog.at_level(WARNING):
        matrix = build_training_matrix(
            vault_path=vault_path,
            pool_path=pool_path,
            index_path=index_path,
            matching_config=_matching_config(),
            pairing_config=_pairing_config(),
        )
    assert matrix.missing_in_pool == 1
    assert "marc_not_in_pool" in caplog.text


def test_build_training_matrix_warns_cce_not_in_index(
    tmp_path: Path,
    caplog: LogCaptureFixture,
) -> None:
    """A vault CCE absent from the index is logged and skipped."""
    index_path = _build_index(tmp_path)
    pool_path = tmp_path / "pool"
    _write_pool(pool_path, "marc-aaa")
    vault_path = tmp_path / "vault.jsonl"
    _seed(vault_path, (_entry("marc-aaa", "UUID-9999", "match", 0),))
    with caplog.at_level(WARNING):
        matrix = build_training_matrix(
            vault_path=vault_path,
            pool_path=pool_path,
            index_path=index_path,
            matching_config=_matching_config(),
            pairing_config=_pairing_config(),
        )
    assert matrix.missing_in_index == 1
    assert "cce_not_in_index" in caplog.text


def test_train_learned_model_rejects_single_class(tmp_path: Path) -> None:
    """Training on a single-class matrix raises a clear error."""
    index_path = _build_index(tmp_path)
    pool_path = tmp_path / "pool"
    _write_pool(pool_path, "marc-aaa")
    vault_path = tmp_path / "vault.jsonl"
    _seed(vault_path, (_entry("marc-aaa", "UUID-0001", "match", 0),))
    matrix = build_training_matrix(
        vault_path=vault_path,
        pool_path=pool_path,
        index_path=index_path,
        matching_config=_matching_config(),
        pairing_config=_pairing_config(),
    )
    with raises(ValueError, match="both match and no_match"):
        train_learned_model(matrix)
