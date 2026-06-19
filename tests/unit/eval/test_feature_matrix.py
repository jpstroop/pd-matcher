"""Tests for :mod:`pd_matcher.eval.feature_matrix`."""

from logging import WARNING
from pathlib import Path

from numpy import float64
from numpy import int64
from pytest import LogCaptureFixture

from pd_groundtruth.label_vault import SCHEMA_VERSION
from pd_groundtruth.label_vault import MarcIdentifiers
from pd_groundtruth.label_vault import VaultEntry
from pd_groundtruth.label_vault import upsert_entry
from pd_matcher.config.loader import load_pairing_config
from pd_matcher.config.schemas import MatchingConfig
from pd_matcher.config.schemas import PairingConfig
from pd_matcher.eval.feature_matrix import SCORER_ORDER
from pd_matcher.eval.feature_matrix import FeatureMatrixRow
from pd_matcher.eval.feature_matrix import extract_feature_matrix
from pd_matcher.eval.feature_matrix import feature_column_names
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


def test_feature_column_names_layout() -> None:
    """Column names are deterministic: scores first, then matching skipped flags."""
    names = feature_column_names()
    expected_scores = list(SCORER_ORDER)
    expected_flags = [f"{scorer}__skipped" for scorer in SCORER_ORDER]
    assert names == expected_scores + expected_flags
    assert len(names) == 2 * len(SCORER_ORDER)


def test_extract_feature_matrix_returns_correct_shapes(tmp_path: Path) -> None:
    """One match + one no_match -> 2x(2*scorers) matrix, both labels present."""
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
                verdict="no_match",
            ),
        ),
    )
    x, y, names, rows = extract_feature_matrix(
        vault_path=vault_path,
        pool_path=pool_path,
        index_path=index_path,
        matching_config=_matching_config(),
        pairing_config=_pairing_config(),
    )
    assert x.shape == (2, 2 * len(SCORER_ORDER))
    assert y.shape == (2,)
    assert x.dtype == float64
    assert y.dtype == int64
    assert names == feature_column_names()
    assert len(rows) == 2
    assert {row.verdict for row in rows} == {"match", "no_match"}
    assert sorted(int(label) for label in y) == [0, 1]


def test_extract_feature_matrix_excludes_unsure(tmp_path: Path) -> None:
    """``unsure`` verdicts never reach the matrix at all."""
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
    x, y, _names, rows = extract_feature_matrix(
        vault_path=vault_path,
        pool_path=pool_path,
        index_path=index_path,
        matching_config=_matching_config(),
        pairing_config=_pairing_config(),
    )
    assert x.shape == (1, 2 * len(SCORER_ORDER))
    assert y.shape == (1,)
    assert rows[0].verdict == "match"
    assert int(y[0]) == 1


def test_extract_feature_matrix_logs_marc_not_in_pool(
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
                marc_control_id="marc-missing",
                nypl_uuid="UUID-0001",
                verdict="match",
            ),
        ),
    )
    with caplog.at_level(WARNING, logger="pd_matcher.eval.feature_matrix"):
        x, y, _names, rows = extract_feature_matrix(
            vault_path=vault_path,
            pool_path=pool_path,
            index_path=index_path,
            matching_config=_matching_config(),
            pairing_config=_pairing_config(),
        )
    assert x.shape == (0, 2 * len(SCORER_ORDER))
    assert y.shape == (0,)
    assert rows == ()
    assert any("marc_not_in_pool" in record.message for record in caplog.records)


def test_extract_feature_matrix_logs_cce_not_in_index(
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
    with caplog.at_level(WARNING, logger="pd_matcher.eval.feature_matrix"):
        x, y, _names, rows = extract_feature_matrix(
            vault_path=vault_path,
            pool_path=pool_path,
            index_path=index_path,
            matching_config=_matching_config(),
            pairing_config=_pairing_config(),
        )
    assert x.shape == (0, 2 * len(SCORER_ORDER))
    assert y.shape == (0,)
    assert rows == ()
    assert any("cce_not_in_index" in record.message for record in caplog.records)


def test_extract_feature_matrix_handles_empty_vault(tmp_path: Path) -> None:
    """An empty vault yields zero-row matrix, not a divide-by-zero."""
    index_path = _build_index(tmp_path)
    pool_path = tmp_path / "pool"
    _write_pool(pool_path, ())
    vault_path = tmp_path / "vault.jsonl"
    vault_path.write_text("", encoding="utf-8")
    x, y, names, rows = extract_feature_matrix(
        vault_path=vault_path,
        pool_path=pool_path,
        index_path=index_path,
        matching_config=_matching_config(),
        pairing_config=_pairing_config(),
    )
    assert x.shape == (0, 2 * len(SCORER_ORDER))
    assert y.shape == (0,)
    assert rows == ()
    assert names == feature_column_names()


def test_extract_feature_matrix_row_carries_match_evidence(tmp_path: Path) -> None:
    """A clean match row carries non-zero title evidence and the matching combined score."""
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
        ),
    )
    x, _y, names, rows = extract_feature_matrix(
        vault_path=vault_path,
        pool_path=pool_path,
        index_path=index_path,
        matching_config=_matching_config(),
        pairing_config=_pairing_config(),
    )
    assert len(rows) == 1
    row = rows[0]
    assert isinstance(row, FeatureMatrixRow)
    assert row.marc_control_id == "marc-aaa"
    assert row.nypl_uuid == "UUID-0001"
    assert 0.0 <= row.combined_score <= 1.0
    title_index = names.index("title.token_set")
    assert float(x[0, title_index]) > 0.0
    assert row.feature_values[title_index] == float(x[0, title_index])
    assert row.marc_title
    assert row.cce_title
