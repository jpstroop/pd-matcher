"""Tests for :mod:`pd_matcher.normalize.claimants`."""

from pd_matcher.normalize.claimants import author_claimant_name
from pd_matcher.normalize.claimants import claimant_class_indicators
from pd_matcher.normalize.claimants import claimant_renewal_label
from pd_matcher.normalize.claimants import parse_claimants


def test_parse_claimants_none_returns_empty() -> None:
    assert parse_claimants(None) == ()


def test_parse_claimants_blank_returns_empty() -> None:
    assert parse_claimants("   ") == ()


def test_parse_claimants_single_name_and_code() -> None:
    assert parse_claimants("Jane Doe|A") == (("Jane Doe", "A"),)


def test_parse_claimants_multi_with_double_pipe() -> None:
    assert parse_claimants("Jane Doe|A||Acme Press|PWH") == (
        ("Jane Doe", "A"),
        ("Acme Press", "PWH"),
    )


def test_parse_claimants_semicolon_separator() -> None:
    assert parse_claimants("Jane Doe|A;John Doe|W") == (
        ("Jane Doe", "A"),
        ("John Doe", "W"),
    )


def test_parse_claimants_multichar_estate_code() -> None:
    assert parse_claimants("Estate of Doe|NK") == (("Estate of Doe", "NK"),)


def test_parse_claimants_strips_surrounding_whitespace_in_code() -> None:
    assert parse_claimants("Jane Doe | A ") == (("Jane Doe", "A"),)


def test_parse_claimants_parenthetical_code_fallback() -> None:
    assert parse_claimants("Jane Doe (PWH)") == (("Jane Doe", "PWH"),)


def test_parse_claimants_skips_blank_trailing_part() -> None:
    assert parse_claimants("Jane Doe|A||") == (("Jane Doe", "A"),)


def test_parse_claimants_no_recognizable_code_keeps_whole_part() -> None:
    assert parse_claimants("Just A Name") == (("Just A Name", ""),)


def test_parse_claimants_lowercase_pipe_code_is_not_a_code() -> None:
    assert parse_claimants("Jane Doe|a") == (("Jane Doe|a", ""),)


def test_parse_claimants_lowercase_parenthetical_is_not_a_code() -> None:
    assert parse_claimants("Jane Doe (a)") == (("Jane Doe (a)", ""),)


def test_claimant_class_indicators_author() -> None:
    assert claimant_class_indicators((("Jane Doe", "A"),)) == (True, False, False)


def test_claimant_class_indicators_estate() -> None:
    assert claimant_class_indicators((("John Doe", "W"),)) == (False, True, False)


def test_claimant_class_indicators_proprietor() -> None:
    assert claimant_class_indicators((("Acme Press", "PWH"),)) == (False, False, True)


def test_claimant_class_indicators_none_when_no_recognized_code() -> None:
    assert claimant_class_indicators((("Just A Name", ""),)) == (False, False, False)


def test_claimant_class_indicators_are_independent() -> None:
    pairs = (("Jane Doe", "A"), ("John Doe", "C"), ("Acme Press", "PCW"))
    assert claimant_class_indicators(pairs) == (True, True, True)


def test_author_claimant_name_returns_first_author() -> None:
    pairs = (("Jane Doe", "A"), ("Second Author", "A"))
    assert author_claimant_name(pairs) == "Jane Doe"


def test_author_claimant_name_skips_non_author_claimants() -> None:
    pairs = (("John Doe", "W"), ("Jane Doe", "A"))
    assert author_claimant_name(pairs) == "Jane Doe"


def test_author_claimant_name_none_when_no_author() -> None:
    assert author_claimant_name((("John Doe", "W"),)) is None


def test_author_claimant_name_none_for_empty() -> None:
    assert author_claimant_name(()) is None


def test_claimant_renewal_label_author() -> None:
    assert claimant_renewal_label("Jane Doe|A") == "author"


def test_claimant_renewal_label_widow() -> None:
    assert claimant_renewal_label("John Doe|W") == "widow/widower (estate)"


def test_claimant_renewal_label_child() -> None:
    assert claimant_renewal_label("Rhoda F. Haynes|C") == "child (estate)"


def test_claimant_renewal_label_executor() -> None:
    assert claimant_renewal_label("Estate of Doe|E") == "executor (estate)"


def test_claimant_renewal_label_next_of_kin() -> None:
    assert claimant_renewal_label("Estate of Doe|NK") == "next of kin (estate)"


def test_claimant_renewal_label_proprietor_work_for_hire() -> None:
    assert claimant_renewal_label("Acme Press|PWH") == "proprietor (work for hire)"


def test_claimant_renewal_label_proprietor_posthumous() -> None:
    assert claimant_renewal_label("Acme Press|PPW") == "proprietor (posthumous work)"


def test_claimant_renewal_label_proprietor_composite() -> None:
    assert claimant_renewal_label("Acme Press|PCW") == "proprietor (composite work)"


def test_claimant_renewal_label_multiple_distinct_classes_joined() -> None:
    assert (
        claimant_renewal_label("Jane Doe|A||Acme Press|PWH") == "author; proprietor (work for hire)"
    )


def test_claimant_renewal_label_deduplicates_repeated_relationship() -> None:
    assert claimant_renewal_label("Jane Doe|C||John Doe|C") == "child (estate)"


def test_claimant_renewal_label_parenthetical_code_form() -> None:
    assert claimant_renewal_label("Jane Doe (PWH)") == "proprietor (work for hire)"


def test_claimant_renewal_label_unrecognized_code_is_unknown() -> None:
    assert claimant_renewal_label("Just A Name") == "Unknown"


def test_claimant_renewal_label_empty_is_unknown() -> None:
    assert claimant_renewal_label("   ") == "Unknown"


def test_claimant_renewal_label_none_is_unknown() -> None:
    assert claimant_renewal_label(None) == "Unknown"
