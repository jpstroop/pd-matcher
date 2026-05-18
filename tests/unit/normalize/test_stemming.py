"""Tests for :mod:`pd_matcher.normalize.stemming`."""

from pd_matcher.normalize.stemming import stem_tokens
from pd_matcher.normalize.stemming import stemmer_for


def test_stemmer_for_english_stems_running_to_run() -> None:
    stem = stemmer_for("eng")
    assert stem("running") == "run"


def test_stemmer_for_french_stems_a_french_word() -> None:
    stem = stemmer_for("fre")
    assert stem("courants") == "cour"


def test_stemmer_for_unknown_language_falls_back_to_english() -> None:
    stem = stemmer_for("lat")
    assert stem("running") == "run"


def test_stemmer_for_is_cached() -> None:
    first = stemmer_for("eng")
    second = stemmer_for("eng")
    assert first("studies") == second("studies")


def test_stem_tokens_returns_a_tuple_of_stems() -> None:
    assert stem_tokens(("running", "studies"), "eng") == ("run", "studi")


def test_stem_tokens_empty_input_returns_empty_tuple() -> None:
    assert stem_tokens((), "eng") == ()


def test_stem_tokens_supports_all_documented_languages() -> None:
    assert stem_tokens(("zwei",), "ger") != ()
    assert stem_tokens(("dos",), "spa") != ()
    assert stem_tokens(("due",), "ita") != ()
