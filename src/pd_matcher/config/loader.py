"""YAML loaders for the configuration schemas defined in :mod:`schemas`.

We keep ``yaml.safe_load`` for parsing (it cleanly separates syntax errors
from schema errors via :class:`yaml.YAMLError`) and feed the resulting
mapping to :func:`msgspec.convert`, which performs constraint validation and
constructs the frozen :class:`~msgspec.Struct` in one step. A small
``dec_hook`` upcasts ``str`` to :class:`pathlib.Path` for fields whose
schema declares them as :class:`Path` (msgspec does not auto-coerce strings
to Path).
"""

from pathlib import Path

from msgspec import ValidationError
from msgspec import convert
from yaml import YAMLError
from yaml import safe_load

from pd_matcher.config.schemas import CopyrightRuleSet
from pd_matcher.config.schemas import IndexConfig
from pd_matcher.config.schemas import MatchingConfig


class ConfigError(Exception):
    """Raised when a configuration file is missing, malformed, or invalid."""


def _path_dec_hook(type_: type, obj: object) -> object:
    """msgspec decode hook upcasting strings to :class:`pathlib.Path`."""
    if type_ is Path and isinstance(obj, str):
        return Path(obj)
    raise NotImplementedError(f"Unsupported decode: {type_!r} from {obj!r}")


def _read_yaml(path: Path) -> object:
    """Read and parse a YAML file, raising :class:`ConfigError` on failure.

    Args:
        path: Filesystem path to a YAML document.

    Returns:
        The parsed YAML structure (typically a ``dict``).

    Raises:
        ConfigError: If the file cannot be read or parsed.
    """
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise ConfigError(f"Cannot read config file {path}: {exc}") from exc
    try:
        return safe_load(text)
    except YAMLError as exc:
        raise ConfigError(f"Malformed YAML in {path}: {exc}") from exc


def load_matching_config(path: Path) -> MatchingConfig:
    """Load and validate a :class:`MatchingConfig` from ``path``.

    Args:
        path: YAML file containing matching weights and thresholds.

    Returns:
        A validated :class:`MatchingConfig`.

    Raises:
        ConfigError: If the file cannot be read, parsed, or validated.
    """
    data = _read_yaml(path)
    try:
        return convert(data, type=MatchingConfig, dec_hook=_path_dec_hook)
    except (ValidationError, ValueError) as exc:
        raise ConfigError(f"Invalid matching config in {path}: {exc}") from exc


def load_copyright_rules(path: Path) -> CopyrightRuleSet:
    """Load and validate a :class:`CopyrightRuleSet` from ``path``.

    Args:
        path: YAML file containing the copyright rule set.

    Returns:
        A validated :class:`CopyrightRuleSet`.

    Raises:
        ConfigError: If the file cannot be read, parsed, or validated.
    """
    data = _read_yaml(path)
    try:
        return convert(data, type=CopyrightRuleSet, dec_hook=_path_dec_hook)
    except (ValidationError, ValueError) as exc:
        raise ConfigError(f"Invalid copyright rule set in {path}: {exc}") from exc


def load_index_config(path: Path) -> IndexConfig:
    """Load and validate an :class:`IndexConfig` from ``path``.

    Args:
        path: YAML file describing the LMDB index location and parameters.

    Returns:
        A validated :class:`IndexConfig`.

    Raises:
        ConfigError: If the file cannot be read, parsed, or validated.
    """
    data = _read_yaml(path)
    try:
        return convert(data, type=IndexConfig, dec_hook=_path_dec_hook)
    except (ValidationError, ValueError) as exc:
        raise ConfigError(f"Invalid index config in {path}: {exc}") from exc


__all__ = [
    "ConfigError",
    "load_copyright_rules",
    "load_index_config",
    "load_matching_config",
]
