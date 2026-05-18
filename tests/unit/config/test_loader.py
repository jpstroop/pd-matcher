"""Tests for :mod:`pd_matcher.config.loader`."""

from importlib.resources import as_file
from importlib.resources import files
from pathlib import Path

from pytest import raises

from pd_matcher.config.loader import ConfigError
from pd_matcher.config.loader import load_copyright_rules
from pd_matcher.config.loader import load_index_config
from pd_matcher.config.loader import load_matching_config


def test_load_shipped_matching_defaults() -> None:
    """The packaged ``matching.yaml`` must validate cleanly."""
    resource = files("pd_matcher.config.defaults") / "matching.yaml"
    with as_file(resource) as path:
        cfg = load_matching_config(path)
    assert cfg.title_weight == 0.40
    assert cfg.author_weight == 0.20
    assert cfg.publisher_weight == 0.10
    assert cfg.year_weight == 0.10
    assert cfg.edition_weight == 0.05
    assert cfg.lccn_weight == 0.10
    assert cfg.isbn_weight == 0.05
    assert cfg.year_window == 2
    assert cfg.min_combined_score == 70.0
    assert cfg.scorer == "weighted_mean"


def test_load_shipped_copyright_rules_defaults() -> None:
    """The packaged ``copyright_rules.yaml`` must validate cleanly."""
    resource = files("pd_matcher.config.defaults") / "copyright_rules.yaml"
    with as_file(resource) as path:
        rs = load_copyright_rules(path)
    assert rs.version == "1.0.0"
    assert len(rs.rules) > 0


def test_load_index_config_from_temp_yaml(tmp_path: Path) -> None:
    """A well-formed index config YAML should produce an ``IndexConfig``."""
    yaml_path = tmp_path / "index.yaml"
    yaml_path.write_text(
        "lmdb_path: caches/nypl.lmdb\nmap_size_bytes: 2048\nschema_version: 3\n",
        encoding="utf-8",
    )
    cfg = load_index_config(yaml_path)
    assert cfg.lmdb_path == Path("caches/nypl.lmdb")
    assert cfg.map_size_bytes == 2048
    assert cfg.schema_version == 3


def test_load_matching_config_raises_on_missing_file(tmp_path: Path) -> None:
    """A nonexistent path must raise :class:`ConfigError`."""
    with raises(ConfigError, match="Cannot read"):
        load_matching_config(tmp_path / "nope.yaml")


def test_load_matching_config_raises_on_malformed_yaml(tmp_path: Path) -> None:
    """Bad YAML syntax should be wrapped in :class:`ConfigError`."""
    bad = tmp_path / "bad.yaml"
    bad.write_text("title_weight: [unterminated\n", encoding="utf-8")
    with raises(ConfigError, match="Malformed YAML"):
        load_matching_config(bad)


def test_load_matching_config_raises_on_schema_violation(tmp_path: Path) -> None:
    """A schema-violating YAML should raise :class:`ConfigError`."""
    bad = tmp_path / "bad_schema.yaml"
    bad.write_text(
        "title_weight: 0.9\nauthor_weight: 0.3\npublisher_weight: 0.1\n"
        "year_weight: 0.05\nedition_weight: 0.05\n"
        "year_window: 2\nmin_combined_score: 50.0\n",
        encoding="utf-8",
    )
    with raises(ConfigError, match="Invalid matching config"):
        load_matching_config(bad)


def test_load_copyright_rules_raises_on_schema_violation(tmp_path: Path) -> None:
    """A schema-violating copyright YAML should raise :class:`ConfigError`."""
    bad = tmp_path / "bad_rules.yaml"
    bad.write_text("rules: []\n", encoding="utf-8")
    with raises(ConfigError, match="Invalid copyright rule set"):
        load_copyright_rules(bad)


def test_load_copyright_rules_raises_on_malformed_yaml(tmp_path: Path) -> None:
    """Bad YAML inside the copyright loader should be wrapped."""
    bad = tmp_path / "bad.yaml"
    bad.write_text("version: 1.0\nrules: [unterminated\n", encoding="utf-8")
    with raises(ConfigError, match="Malformed YAML"):
        load_copyright_rules(bad)


def test_load_index_config_raises_on_schema_violation(tmp_path: Path) -> None:
    """A schema-violating index YAML should raise :class:`ConfigError`."""
    bad = tmp_path / "bad_index.yaml"
    bad.write_text("map_size_bytes: 1024\n", encoding="utf-8")
    with raises(ConfigError, match="Invalid index config"):
        load_index_config(bad)


def test_load_index_config_raises_on_missing_file(tmp_path: Path) -> None:
    """A nonexistent path through ``load_index_config`` must raise."""
    with raises(ConfigError, match="Cannot read"):
        load_index_config(tmp_path / "missing.yaml")


def test_load_copyright_rules_raises_on_missing_file(tmp_path: Path) -> None:
    """A nonexistent path through ``load_copyright_rules`` must raise."""
    with raises(ConfigError, match="Cannot read"):
        load_copyright_rules(tmp_path / "missing.yaml")
