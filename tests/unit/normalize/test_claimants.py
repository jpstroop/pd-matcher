"""Tests for :mod:`pd_matcher.normalize.claimants`."""

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
