"""Number, ordinal, and abbreviation normalization for matcher tokenization.

Three independent operations are exposed for callers that want fine-grained
control, plus a composite :func:`normalize_numbers` that applies all of them
in sequence. The composite is what the matching pipeline uses; the granular
functions are kept public both for unit testing and for the Phase 8 property
tests around Roman numeral round-tripping.

Language codes follow the 3-letter MARC 008 convention (``eng``, ``fre``,
``ger``, ``spa``, ``ita``). Any other code falls back to English tables.
"""

from collections.abc import Mapping
from re import IGNORECASE
from re import Match
from re import compile as re_compile

_ROMAN_PATTERN = re_compile(r"^[ivxlcdm]+$", IGNORECASE)
_ROMAN_VALUES: Mapping[str, int] = {
    "i": 1,
    "v": 5,
    "x": 10,
    "l": 50,
    "c": 100,
    "d": 500,
    "m": 1000,
}

_ENGLISH_NUMBERS: Mapping[str, int] = {
    "zero": 0,
    "one": 1,
    "two": 2,
    "three": 3,
    "four": 4,
    "five": 5,
    "six": 6,
    "seven": 7,
    "eight": 8,
    "nine": 9,
    "ten": 10,
    "eleven": 11,
    "twelve": 12,
    "thirteen": 13,
    "fourteen": 14,
    "fifteen": 15,
    "sixteen": 16,
    "seventeen": 17,
    "eighteen": 18,
    "nineteen": 19,
    "twenty": 20,
    "thirty": 30,
    "forty": 40,
    "fifty": 50,
    "sixty": 60,
    "seventy": 70,
    "eighty": 80,
    "ninety": 90,
    "hundred": 100,
    "thousand": 1000,
}

_FRENCH_NUMBERS: Mapping[str, int] = {
    "zero": 0,
    "un": 1,
    "une": 1,
    "deux": 2,
    "trois": 3,
    "quatre": 4,
    "cinq": 5,
    "six": 6,
    "sept": 7,
    "huit": 8,
    "neuf": 9,
    "dix": 10,
    "onze": 11,
    "douze": 12,
    "treize": 13,
    "quatorze": 14,
    "quinze": 15,
    "seize": 16,
    "vingt": 20,
    "trente": 30,
    "quarante": 40,
    "cinquante": 50,
    "soixante": 60,
    "cent": 100,
    "mille": 1000,
}

_GERMAN_NUMBERS: Mapping[str, int] = {
    "null": 0,
    "eins": 1,
    "ein": 1,
    "eine": 1,
    "zwei": 2,
    "drei": 3,
    "vier": 4,
    "funf": 5,
    "sechs": 6,
    "sieben": 7,
    "acht": 8,
    "neun": 9,
    "zehn": 10,
    "elf": 11,
    "zwolf": 12,
    "dreizehn": 13,
    "vierzehn": 14,
    "funfzehn": 15,
    "sechzehn": 16,
    "siebzehn": 17,
    "achtzehn": 18,
    "neunzehn": 19,
    "zwanzig": 20,
    "dreissig": 30,
    "vierzig": 40,
    "funfzig": 50,
    "sechzig": 60,
    "siebzig": 70,
    "achtzig": 80,
    "neunzig": 90,
    "hundert": 100,
    "tausend": 1000,
}

_SPANISH_NUMBERS: Mapping[str, int] = {
    "cero": 0,
    "uno": 1,
    "una": 1,
    "dos": 2,
    "tres": 3,
    "cuatro": 4,
    "cinco": 5,
    "seis": 6,
    "siete": 7,
    "ocho": 8,
    "nueve": 9,
    "diez": 10,
    "once": 11,
    "doce": 12,
    "trece": 13,
    "catorce": 14,
    "quince": 15,
    "dieciseis": 16,
    "diecisiete": 17,
    "dieciocho": 18,
    "diecinueve": 19,
    "veinte": 20,
    "treinta": 30,
    "cuarenta": 40,
    "cincuenta": 50,
    "sesenta": 60,
    "setenta": 70,
    "ochenta": 80,
    "noventa": 90,
    "cien": 100,
    "mil": 1000,
}

_ITALIAN_NUMBERS: Mapping[str, int] = {
    "zero": 0,
    "uno": 1,
    "una": 1,
    "due": 2,
    "tre": 3,
    "quattro": 4,
    "cinque": 5,
    "sei": 6,
    "sette": 7,
    "otto": 8,
    "nove": 9,
    "dieci": 10,
    "undici": 11,
    "dodici": 12,
    "tredici": 13,
    "quattordici": 14,
    "quindici": 15,
    "sedici": 16,
    "diciassette": 17,
    "diciotto": 18,
    "diciannove": 19,
    "venti": 20,
    "trenta": 30,
    "quaranta": 40,
    "cinquanta": 50,
    "sessanta": 60,
    "settanta": 70,
    "ottanta": 80,
    "novanta": 90,
    "cento": 100,
    "mille": 1000,
}

_ENGLISH_ORDINALS: Mapping[str, int] = {
    "first": 1,
    "second": 2,
    "third": 3,
    "fourth": 4,
    "fifth": 5,
    "sixth": 6,
    "seventh": 7,
    "eighth": 8,
    "ninth": 9,
    "tenth": 10,
    "eleventh": 11,
    "twelfth": 12,
    "thirteenth": 13,
    "fourteenth": 14,
    "fifteenth": 15,
    "sixteenth": 16,
    "seventeenth": 17,
    "eighteenth": 18,
    "nineteenth": 19,
    "twentieth": 20,
}

_FRENCH_ORDINALS: Mapping[str, int] = {
    "premier": 1,
    "premiere": 1,
    "deuxieme": 2,
    "second": 2,
    "seconde": 2,
    "troisieme": 3,
    "quatrieme": 4,
    "cinquieme": 5,
    "sixieme": 6,
    "septieme": 7,
    "huitieme": 8,
    "neuvieme": 9,
    "dixieme": 10,
}

_GERMAN_ORDINALS: Mapping[str, int] = {
    "erste": 1,
    "erster": 1,
    "ersten": 1,
    "zweite": 2,
    "zweiter": 2,
    "dritte": 3,
    "dritter": 3,
    "vierte": 4,
    "funfte": 5,
    "sechste": 6,
    "siebte": 7,
    "achte": 8,
    "neunte": 9,
    "zehnte": 10,
}

_SPANISH_ORDINALS: Mapping[str, int] = {
    "primero": 1,
    "primer": 1,
    "primera": 1,
    "segundo": 2,
    "segunda": 2,
    "tercero": 3,
    "tercer": 3,
    "tercera": 3,
    "cuarto": 4,
    "cuarta": 4,
    "quinto": 5,
    "quinta": 5,
    "sexto": 6,
    "septimo": 7,
    "octavo": 8,
    "noveno": 9,
    "decimo": 10,
}

_ITALIAN_ORDINALS: Mapping[str, int] = {
    "primo": 1,
    "prima": 1,
    "secondo": 2,
    "seconda": 2,
    "terzo": 3,
    "terza": 3,
    "quarto": 4,
    "quarta": 4,
    "quinto": 5,
    "quinta": 5,
    "sesto": 6,
    "settimo": 7,
    "ottavo": 8,
    "nono": 9,
    "decimo": 10,
}

_NUMBER_TABLES: Mapping[str, Mapping[str, int]] = {
    "eng": _ENGLISH_NUMBERS,
    "fre": _FRENCH_NUMBERS,
    "ger": _GERMAN_NUMBERS,
    "spa": _SPANISH_NUMBERS,
    "ita": _ITALIAN_NUMBERS,
}

_ORDINAL_TABLES: Mapping[str, Mapping[str, int]] = {
    "eng": _ENGLISH_ORDINALS,
    "fre": _FRENCH_ORDINALS,
    "ger": _GERMAN_ORDINALS,
    "spa": _SPANISH_ORDINALS,
    "ita": _ITALIAN_ORDINALS,
}

_ABBREVIATIONS: Mapping[str, str] = {
    "v": "volume",
    "vol": "volume",
    "vols": "volume",
    "ed": "edition",
    "no": "number",
    "nos": "number",
    "pt": "part",
    "pts": "part",
    "bk": "book",
    "inc": "incorporated",
    "corp": "corporation",
    "co": "company",
    "bros": "brothers",
    "ltd": "limited",
    "pub": "publishing",
    "pubs": "publishing",
    "publ": "publishing",
    "soc": "society",
    "assn": "association",
    "assoc": "association",
    "univ": "university",
    "calif": "california",
}

_ABBREVIATION_RE = re_compile(
    r"\b(" + "|".join(sorted(_ABBREVIATIONS, key=len, reverse=True)) + r")\.",
    IGNORECASE,
)

_PUNCT_STRIP = ".,;:!?\"'()[]{}/"


def roman_to_arabic(roman: str) -> int | None:
    """Parse a Roman numeral string (case-insensitive) to its integer value.

    Args:
        roman: Candidate Roman numeral, e.g. ``"XIV"`` or ``"mcm"``.

    Returns:
        The integer value, or ``None`` if ``roman`` is empty or contains
        characters outside ``ivxlcdm``.
    """
    if not roman or not _ROMAN_PATTERN.match(roman):
        return None
    lowered = roman.lower()
    total = 0
    previous = 0
    for char in reversed(lowered):
        value = _ROMAN_VALUES[char]
        if value < previous:
            total -= value
        else:
            total += value
            previous = value
    return total


def _table_for(tables: Mapping[str, Mapping[str, int]], language: str) -> Mapping[str, int]:
    """Return the language-specific table, falling back to English."""
    return tables.get(language, tables["eng"])


def word_to_int(word: str, language: str) -> int | None:
    """Translate a number word ("three", "drei") to an ``int``.

    Args:
        word: Candidate number word; matched case-insensitively.
        language: MARC 3-letter language code; unknown codes fall back
            to English.

    Returns:
        The integer value, or ``None`` if ``word`` is not in the table.
    """
    if not word:
        return None
    table = _table_for(_NUMBER_TABLES, language)
    return table.get(word.lower())


def ordinal_word_to_int(word: str, language: str) -> int | None:
    """Translate an ordinal word ("first", "premier") to an ``int``.

    Args:
        word: Candidate ordinal word; matched case-insensitively.
        language: MARC 3-letter language code; unknown codes fall back
            to English.

    Returns:
        The integer value, or ``None`` if ``word`` is not in the table.
    """
    if not word:
        return None
    table = _table_for(_ORDINAL_TABLES, language)
    return table.get(word.lower())


def _expand_abbreviations(text: str) -> str:
    """Replace bibliographic abbreviations with their long form."""

    def _replace(match: Match[str]) -> str:
        return _ABBREVIATIONS[match.group(1).lower()]

    return _ABBREVIATION_RE.sub(_replace, text)


def normalize_numbers(s: str, language: str) -> str:
    """Apply Roman/word/ordinal conversion and abbreviation expansion.

    The function operates token-by-token after expanding documented
    abbreviations so that each transformed token is independent. Tokens that
    do not match any table pass through unchanged.

    Args:
        s: Input text (typically a title or statement of responsibility).
        language: MARC 3-letter language code; unknown codes fall back to
            English tables.

    Returns:
        The text with numbers normalized into Arabic digits.
    """
    if not s:
        return s
    expanded = _expand_abbreviations(s)
    tokens = expanded.split()
    out: list[str] = []
    for token in tokens:
        core = token.strip(_PUNCT_STRIP)
        if not core:
            out.append(token)
            continue
        ordinal = ordinal_word_to_int(core, language)
        if ordinal is not None:
            out.append(str(ordinal))
            continue
        number = word_to_int(core, language)
        if number is not None:
            out.append(str(number))
            continue
        roman = roman_to_arabic(core)
        if roman is not None:
            out.append(str(roman))
            continue
        out.append(token)
    return " ".join(out)


__all__ = [
    "normalize_numbers",
    "ordinal_word_to_int",
    "roman_to_arabic",
    "word_to_int",
]
