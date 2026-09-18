"""Tests for src/text_bn.py - one test per rule, each with a plain-language name.

Run from the repository root:

    python -m pytest tests/test_text_bn.py -v

If a teacher asks for a change (a new stop word, a new protected word, a new
stemming ending), make the change, then run this file and `python src/text_bn.py --check`.
A failing test names the rule that the change broke.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from text_bn import (  # noqa: E402  (the import has to come after the path is set)
    VARIANTS,
    clean,
    preprocess,
    protected_words,
    remove_stopwords,
    stem_word,
    stopwords,
    suffixes,
    tokenize,
)

# A tiny list of "real words" so the stemming tests do not depend on the data.
KNOWN = frozenset({"কলেজ", "ছাত্র", "শব্দ", "বই", "মন্দির", "নেতা", "সাল"})


# ---------------------------------------------------------------- cleaning ---

def test_citation_marks_are_removed():
    assert clean("হয়।[1][২]") == "হয়।"


def test_latex_roots_are_kept():
    assert clean(r"\sqrt[7]{n^3}") == r"\sqrt[7]{n^3}"


def test_bengali_digits_become_english_digits_with_the_same_value():
    assert clean("১৭০৪ সালে") == "1704 সালে"


def test_whitespace_is_collapsed():
    assert clean("  এক\n\tদুই   তিন ") == "এক দুই তিন"


def test_both_spellings_of_ya_become_identical():
    one_code_point = "নয়"          # য + ়   (two code points)
    other_form = "নয়"               # য়      (one code point)
    assert clean(one_code_point) == clean(other_form)


def test_cleaning_twice_changes_nothing():
    text = "  ১৯৭১ সালে[১] স্বাধীন হয়। "
    assert clean(clean(text)) == clean(text)


def test_empty_text_gives_empty_string():
    assert clean(None) == "" and clean("") == ""


# -------------------------------------------------------------- tokenizing ---

def test_bengali_full_stop_is_its_own_token():
    assert tokenize("স্বাধীন হয়।") == ["স্বাধীন", "হয়", "।"]


def test_numbers_with_decimals_stay_one_token():
    assert tokenize("মান 3.14 এবং 1,000") == ["মান", "3.14", "এবং", "1,000"]


def test_tokenizing_loses_nothing_but_spaces():
    text = clean("$f(n) = \\sqrt[7]{n^3}$ হলে বিসিএস-এর ফল কী?")
    assert "".join(tokenize(text)) == text.replace(" ", "")


# -------------------------------------------------------------- word lists ---

def test_negation_and_number_words_are_protected():
    for word in ["না", "নয়", "নেই", "নি", "একটি", "দুটি", "প্রথম"]:
        assert clean(word) in protected_words(), word


def test_published_stop_list_really_contains_negation():
    # This is WHY protection exists. If the list ever changes, re-read the PRD §5.3c.
    assert clean("না") in stopwords() and clean("নয়") in stopwords()


def test_suffixes_are_tried_longest_first():
    lengths = [len(s) for s in suffixes()]
    assert lengths == sorted(lengths, reverse=True)


# -------------------------------------------------------- stop-word removal ---

def test_stopword_removal_keeps_negation():
    tokens = tokenize(clean("বিভাগ আবার বিভক্ত নয়"))
    assert "নয়" in remove_stopwords(tokens)


def test_demo_variant_shows_the_negation_being_lost():
    tokens = tokenize(clean("বিভাগ আবার বিভক্ত নয়"))
    assert clean("নয়") not in remove_stopwords(tokens, keep_protected=False)


# ---------------------------------------------------------------- stemming ---

def test_noun_endings_are_cut():
    assert stem_word("কলেজের", KNOWN) == "কলেজ"
    assert stem_word("ছাত্ররা", KNOWN) == "ছাত্র"
    assert stem_word("নেতাদের", KNOWN) == "নেতা"
    assert stem_word("মন্দিরের", KNOWN) == "মন্দির"


def test_cut_must_leave_a_real_word():
    # "দের" is tried first but leaves the non-word "শব্"; "ের" leaves "শব্দ".
    assert stem_word("শব্দের", KNOWN) == "শব্দ"


def test_unknown_stem_leaves_the_word_unchanged():
    assert stem_word("অজানাশব্দের", KNOWN) == "অজানাশব্দের"


def test_short_stems_are_not_cut():
    assert stem_word("ধারা", KNOWN | {"ধা"}) == "ধারা"


def test_negation_endings_are_never_cut():
    assert stem_word("করেননি", KNOWN | {"করেনন", "করেন"}) == "করেননি"
    assert stem_word("পড়েনি", KNOWN) == "পড়েনি"


def test_protected_words_are_never_cut():
    assert stem_word(clean("দুটি"), KNOWN | {"দু"}) == clean("দুটি")


def test_numbers_and_english_are_never_cut():
    assert stem_word("1971", KNOWN) == "1971"
    assert stem_word("DNA", KNOWN) == "DNA"


# ---------------------------------------------------------------- variants ---

def test_v0_is_plain_space_splitting():
    assert preprocess("১৯৭১ সালে[১]", "V0") == ["১৯৭১", "সালে[১]"]


def test_v1_is_clean_plus_tokenize():
    text = "১৯৭১ সালে[১] স্বাধীন হয়।"
    assert preprocess(text, "V1") == tokenize(clean(text))


def test_every_normal_variant_keeps_negation():
    text = "সব বিভাগ আবার সাতটি অঞ্চলে বিভক্ত নয়।"
    for variant in ("V1", "V2", "V3", "V4"):
        assert clean("নয়") in preprocess(text, variant), variant


def test_unknown_variant_is_an_error():
    try:
        preprocess("কিছু", "V9")
    except ValueError:
        return
    raise AssertionError("an unknown variant must raise ValueError")


def test_all_six_variants_exist():
    assert list(VARIANTS) == ["V0", "V1", "V2", "V3", "V4", "V2-demo"]
