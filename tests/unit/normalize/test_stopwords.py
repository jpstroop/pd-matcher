"""Tests for :mod:`pd_matcher.normalize.stopwords`."""

from pd_matcher.normalize.stopwords import StopwordSet
from pd_matcher.normalize.stopwords import load_stopwords


def test_load_stopwords_returns_frozensets_for_english() -> None:
    sw = load_stopwords("eng")
    assert isinstance(sw, StopwordSet)
    assert "the" in sw.title
    assert "and" in sw.author
    assert "of" in sw.publisher


def test_load_stopwords_caches_results() -> None:
    first = load_stopwords("eng")
    second = load_stopwords("eng")
    assert first is second


def test_load_stopwords_supports_each_documented_language() -> None:
    for code in ("eng", "fre", "ger", "spa", "ita"):
        sw = load_stopwords(code)
        assert isinstance(sw.title, frozenset)


def test_load_stopwords_falls_back_to_english_for_unknown_language() -> None:
    fallback = load_stopwords("zzz")
    english = load_stopwords("eng")
    assert fallback is english


def test_stopword_set_is_frozen() -> None:
    sw = load_stopwords("eng")
    try:
        setattr(sw, "title", frozenset())
    except AttributeError:
        return
    raise AssertionError("StopwordSet should reject mutation")


_ENGLISH_PUBLISHER_NOISE_ADDITIONS: frozenset[str] = frozenset(
    {
        "&",
        "co",
        "company",
        "inc",
        "incorporated",
        "corp",
        "corporation",
        "bros",
        "brothers",
        "ltd",
        "limited",
        "pub",
        "publ",
        "pubs",
        "publishing",
        "publisher",
        "publishers",
        "press",
        "soc",
        "society",
        "assn",
        "assoc",
        "association",
        "book",
        "books",
    }
)


_ENGLISH_PUBLISHER_INSTITUTIONAL_ADDITIONS: frozenset[str] = frozenset(
    {
        "university",
        "institute",
        "institution",
        "academy",
        "national",
        "bureau",
        "department",
        "dept",
        "division",
        "office",
        "committee",
        "council",
        "foundation",
        "museum",
        "library",
        "united",
        "nations",
        "international",
        "commission",
    }
)


# The inter-governmental subset of the institutional batch is non-distinguishing
# on BOTH the publisher and author sides ('United Nations' is a corporate author),
# so it is stopworded in both; the rest of the batch ('university', 'bureau', ...)
# stays publisher-only because those are not author tokens.
_ENGLISH_INTERGOV_ADDITIONS: frozenset[str] = frozenset(
    {
        "united",
        "nations",
        "international",
        "commission",
    }
)


def test_english_publisher_stopwords_include_publisher_noise() -> None:
    publisher = load_stopwords("eng").publisher
    assert _ENGLISH_PUBLISHER_NOISE_ADDITIONS.issubset(publisher)


def test_english_publisher_stopwords_include_institutional_generics() -> None:
    publisher = load_stopwords("eng").publisher
    assert _ENGLISH_PUBLISHER_INSTITUTIONAL_ADDITIONS.issubset(publisher)


def test_english_author_stopwords_include_intergovernmental_generics() -> None:
    author = load_stopwords("eng").author
    assert _ENGLISH_INTERGOV_ADDITIONS.issubset(author)


def test_institutional_additions_are_not_english_title_stopwords() -> None:
    english = load_stopwords("eng")
    assert _ENGLISH_PUBLISHER_INSTITUTIONAL_ADDITIONS.isdisjoint(english.title)


def test_publisher_only_institutional_additions_are_not_english_author_stopwords() -> None:
    english = load_stopwords("eng")
    publisher_only = _ENGLISH_PUBLISHER_INSTITUTIONAL_ADDITIONS - _ENGLISH_INTERGOV_ADDITIONS
    assert publisher_only.isdisjoint(english.author)


def test_publisher_noise_additions_are_not_english_title_stopwords() -> None:
    title = load_stopwords("eng").title
    promotable = _ENGLISH_PUBLISHER_NOISE_ADDITIONS - {"&"}
    assert promotable.isdisjoint(title), (
        "Publisher-side noise must not leak into title stopwords; doing so"
        " would strip distinguishing tokens from names like 'Penguin Books'."
    )


def test_publisher_noise_additions_are_not_english_author_stopwords() -> None:
    author = load_stopwords("eng").author
    promotable = _ENGLISH_PUBLISHER_NOISE_ADDITIONS - {"&"}
    assert promotable.isdisjoint(author)
