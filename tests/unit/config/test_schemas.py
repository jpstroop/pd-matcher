"""Tests for :mod:`pd_matcher.config.schemas`."""

from pathlib import Path

from msgspec import ValidationError
from msgspec import convert
from msgspec import to_builtins
from pytest import raises

from pd_matcher.config.loader import _path_dec_hook
from pd_matcher.config.schemas import CopyrightRule
from pd_matcher.config.schemas import CopyrightRuleSet
from pd_matcher.config.schemas import IndexConfig
from pd_matcher.config.schemas import MatchingConfig


def _valid_matching() -> dict[str, object]:
    return {
        "title_weight": 0.40,
        "author_weight": 0.20,
        "publisher_weight": 0.10,
        "year_weight": 0.10,
        "edition_weight": 0.05,
        "lccn_weight": 0.10,
        "isbn_weight": 0.05,
        "year_window": 2,
        "min_combined_score": 70.0,
        "scorer": "weighted_mean",
    }


def test_matching_config_roundtrip() -> None:
    """Constructing and re-validating should produce an equivalent model."""
    cfg = convert(_valid_matching(), type=MatchingConfig)
    again = convert(to_builtins(cfg), type=MatchingConfig)
    assert cfg == again


def test_matching_config_defaults_scorer_to_weighted_mean() -> None:
    """The ``scorer`` field defaults to ``"weighted_mean"`` when omitted."""
    data = _valid_matching()
    del data["scorer"]
    cfg = convert(data, type=MatchingConfig)
    assert cfg.scorer == "weighted_mean"


def test_matching_config_rejects_weights_not_summing_to_one() -> None:
    """The cross-field validator should reject weight tuples != 1.0."""
    data = _valid_matching()
    data["title_weight"] = 0.7
    with raises(ValueError, match=r"must sum to 1\.0"):
        convert(data, type=MatchingConfig)


def test_matching_config_accepts_weights_within_tolerance() -> None:
    """Weights summing to ~1.0 within the small tolerance must be accepted."""
    data = _valid_matching()
    data["title_weight"] = 0.4005
    data["author_weight"] = 0.1995
    cfg = convert(data, type=MatchingConfig)
    assert cfg.publisher_weight == 0.10


def test_matching_config_rejects_negative_weight() -> None:
    """A negative weight should fail the ``ge=0`` constraint."""
    data = _valid_matching()
    data["title_weight"] = -0.1
    with raises(ValidationError):
        convert(data, type=MatchingConfig)


def test_matching_config_is_frozen() -> None:
    """Frozen structs must reject attribute mutation."""
    cfg = convert(_valid_matching(), type=MatchingConfig)
    with raises(AttributeError):
        setattr(cfg, "title_weight", 0.9)


def test_matching_config_forbids_extra_fields() -> None:
    """Unknown keys must be rejected."""
    data = _valid_matching()
    data["mystery_field"] = True
    with raises(ValidationError):
        convert(data, type=MatchingConfig)


def test_copyright_rule_roundtrip_and_validation() -> None:
    """CopyrightRule should round-trip and reject blank predicates."""
    rule = convert(
        {
            "name": "pd_pre_1929",
            "when": ["published_in_range(0, 1928)"],
            "then": "PD_PRE_1929",
            "explanation": "Anything published before 1929 is in the public domain.",
        },
        type=CopyrightRule,
    )
    again = convert(to_builtins(rule), type=CopyrightRule)
    assert rule == again
    with raises(ValueError, match=r"when\[0\] is empty"):
        convert(
            {
                "name": "bad",
                "when": ["   "],
                "then": "UNKNOWN",
                "explanation": "x",
            },
            type=CopyrightRule,
        )


def test_copyright_rule_is_frozen_and_forbids_extras() -> None:
    """CopyrightRule must reject mutation and unknown fields."""
    rule = convert(
        {"name": "r", "when": [], "then": "X", "explanation": "why"},
        type=CopyrightRule,
    )
    with raises(AttributeError):
        setattr(rule, "name", "other")
    with raises(ValidationError):
        convert(
            {"name": "r", "when": [], "then": "X", "explanation": "why", "x": 1},
            type=CopyrightRule,
        )


def test_copyright_rule_set_roundtrip() -> None:
    """CopyrightRuleSet should round-trip through to_builtins."""
    rs = convert(
        {
            "version": "1.0.0",
            "rules": [
                {"name": "a", "when": [], "then": "X", "explanation": "y"},
            ],
        },
        type=CopyrightRuleSet,
    )
    again = convert(to_builtins(rs), type=CopyrightRuleSet)
    assert rs == again
    assert rs.version == "1.0.0"
    assert len(rs.rules) == 1


def test_copyright_rule_set_defaults_rules_to_empty() -> None:
    """The ``rules`` field defaults to an empty list when omitted."""
    rs = convert({"version": "0.0.1"}, type=CopyrightRuleSet)
    assert rs.rules == []


def test_copyright_rule_set_is_frozen_and_forbids_extras() -> None:
    """CopyrightRuleSet must reject mutation and unknown fields."""
    rs = convert({"version": "1.0.0", "rules": []}, type=CopyrightRuleSet)
    with raises(AttributeError):
        setattr(rs, "version", "2.0.0")
    with raises(ValidationError):
        convert({"version": "1.0.0", "rules": [], "extra": 1}, type=CopyrightRuleSet)


def test_index_config_defaults_and_roundtrip() -> None:
    """IndexConfig should accept defaults and round-trip cleanly."""
    cfg = convert({"lmdb_path": "caches/nypl.lmdb"}, type=IndexConfig, dec_hook=_path_dec_hook)
    assert cfg.lmdb_path == Path("caches/nypl.lmdb")
    assert cfg.map_size_bytes == 16 * 1024 * 1024 * 1024
    assert cfg.schema_version == 1
    again = convert(to_builtins(cfg, enc_hook=str), type=IndexConfig, dec_hook=_path_dec_hook)
    assert again == cfg


def test_index_config_rejects_invalid_sizes() -> None:
    """``map_size_bytes`` and ``schema_version`` must be positive."""
    with raises(ValidationError):
        convert(
            {"lmdb_path": "x", "map_size_bytes": 0},
            type=IndexConfig,
            dec_hook=_path_dec_hook,
        )
    with raises(ValidationError):
        convert(
            {"lmdb_path": "x", "schema_version": 0},
            type=IndexConfig,
            dec_hook=_path_dec_hook,
        )


def test_index_config_is_frozen_and_forbids_extras() -> None:
    """IndexConfig must reject mutation and unknown fields."""
    cfg = convert({"lmdb_path": "x"}, type=IndexConfig, dec_hook=_path_dec_hook)
    with raises(AttributeError):
        setattr(cfg, "schema_version", 99)
    with raises(ValidationError):
        convert({"lmdb_path": "x", "unknown": 1}, type=IndexConfig, dec_hook=_path_dec_hook)


def test_path_dec_hook_rejects_unsupported_types() -> None:
    """The decode hook only handles ``str -> Path``; other inputs must raise."""
    with raises(NotImplementedError):
        _path_dec_hook(Path, 42)
    with raises(NotImplementedError):
        _path_dec_hook(int, "x")
