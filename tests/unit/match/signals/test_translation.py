"""Tests for :mod:`pd_matcher.match.signals.translation`."""

from pd_matcher.match.signals.translation import any_value_matches
from pd_matcher.match.signals.translation import is_translation_signal
from pd_matcher.models import IndexedNyplRegRecord


def _record(
    *,
    desc: str | None = None,
    notes: tuple[str, ...] = (),
    new_matter_claimed: str | None = None,
    renewal_new_matter: str | None = None,
) -> IndexedNyplRegRecord:
    return IndexedNyplRegRecord(
        uuid="u",
        title="t",
        was_renewed=False,
        desc=desc,
        notes=notes,
        new_matter_claimed=new_matter_claimed,
        renewal_new_matter=renewal_new_matter,
    )


def test_tr_abbreviation_fires() -> None:
    """The ``"tr."`` abbreviation triggers the translation signal."""
    assert is_translation_signal(_record(desc="312 p. tr. by R. Smith")) is True


def test_trans_dot_fires() -> None:
    """``"trans."`` (with trailing period) triggers the signal."""
    assert is_translation_signal(_record(desc="trans. from the original")) is True


def test_translated_fires() -> None:
    """``"translated"`` triggers the signal."""
    assert is_translation_signal(_record(desc="translated by John Smith")) is True


def test_translation_fires() -> None:
    """``"translation"`` triggers the signal."""
    assert is_translation_signal(_record(new_matter_claimed="English translation")) is True


def test_bare_version_fires() -> None:
    """``"English version"`` (no ``program``) triggers the signal."""
    assert is_translation_signal(_record(desc="English version of the original")) is True


def test_version_with_program_suppressed() -> None:
    """``"version 2.0 program"`` must NOT fire (computer-software false positive)."""
    assert is_translation_signal(_record(desc="version 2.0 program manual")) is False


def test_from_the_french_fires() -> None:
    """``"from the French"`` triggers the language-named pattern."""
    assert is_translation_signal(_record(desc="Translated from the French")) is True


def test_from_the_yiddish_fires() -> None:
    """``"from the Yiddish"`` triggers the language-named pattern."""
    assert is_translation_signal(_record(desc="From the Yiddish, by S. Goldberg")) is True


def test_from_the_unknown_language_does_not_fire() -> None:
    """A language not in the explicit list does not fire the from-the pattern."""
    assert is_translation_signal(_record(desc="from the Esperanto")) is False


def test_signal_checks_notes_field() -> None:
    """Notes are joined and scanned for translation cues."""
    assert is_translation_signal(_record(notes=("Original work in German",))) is False
    assert is_translation_signal(_record(notes=("Translated from the German",))) is True


def test_signal_checks_renewal_new_matter() -> None:
    """``renewal_new_matter`` is scanned for cues."""
    assert is_translation_signal(_record(renewal_new_matter="English translation")) is True


def test_no_signal_returns_false() -> None:
    """A record with no translation cues yields ``False``."""
    assert is_translation_signal(_record(desc="312 p. illus.")) is False


def test_signal_on_empty_record() -> None:
    """A record with no text fields yields ``False`` (no spurious matches)."""
    assert is_translation_signal(_record()) is False


def test_signal_ignores_unrelated_word_with_tr() -> None:
    """The ``\\b`` word boundary keeps unrelated words from firing."""
    assert is_translation_signal(_record(desc="contracting party")) is False
    assert is_translation_signal(_record(desc="strange travel notes")) is False


def test_any_value_matches_handles_none() -> None:
    """``any_value_matches`` skips ``None`` and empty values."""
    assert any_value_matches(None, "", None) is False


def test_any_value_matches_returns_true_on_first_hit() -> None:
    """``any_value_matches`` returns ``True`` when any value carries a cue."""
    assert any_value_matches(None, "plain text", "translated from the German") is True
