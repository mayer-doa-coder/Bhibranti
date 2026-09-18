"""Bengali text tools: cleaning, tokenizing, stop words, stemming (NLP Lab 1 -> M10).

Every model that is NOT a pretrained encoder (M1-M3, M11-M14) reads its text through
this one file, so all of them see exactly the same words.

    from text_bn import clean, tokenize, preprocess

    clean("১৯৭১ সালে[১] স্বাধীন   হয়।")    ->  "1971 সালে স্বাধীন হয়।"
    tokenize("1971 সালে স্বাধীন হয়।")      ->  ["1971", "সালে", "স্বাধীন", "হয়", "।"]
    preprocess(text)                       ->  the default (V1): clean, then tokenize
    preprocess(text, variant="V4")         ->  clean, tokenize, remove stop words, stem

Run from the repository root:

    python src/text_bn.py "কলেজের ছাত্ররা বইটি পড়েনি।"   # show every step on your own sentence
    python src/text_bn.py --demo                          # show every step on built-in examples
    python src/text_bn.py --check                         # safety checks on the train + dev data

WHY NOT JUST USE THE LAB 1 CODE?
The lab code was written for English. On Bengali it silently breaks the data
(PRD §5.3c):
  * re.sub(r'[^A-Za-z\\s]', '', text) deletes every Bengali letter and every digit.
  * NLTK's word_tokenize, English stop words and PorterStemmer do not know Bengali.
  * A copied stop-word list deletes না / নয় ("not"), which turns an answer into its
    opposite, and deletes number words (একটি / দুটি), which hides numeric errors.
This file keeps the lab's ideas (regex cleaning, tokenizing, stop words, a
Porter-style stemmer) but makes each one safe for Bengali.

WHAT YOU CAN CHANGE WITHOUT TOUCHING THE LOGIC
  * Stop words ................ configs/bn_stopwords.txt      (published list, keep unchanged)
  * Words never removed/cut ... configs/bn_protected_words.txt
  * Endings the stemmer cuts .. configs/bn_suffixes.txt
  * Minimum stem length ....... MIN_STEM_LENGTH below
  * What counts as a real word  KNOWN_WORD_MIN_COUNT below
After any change, run  python src/text_bn.py --check  and  python -m pytest tests/test_text_bn.py
"""

from __future__ import annotations

import argparse
import collections
import re
import sys
import unicodedata
from functools import lru_cache
from pathlib import Path

# =============================================================================
# SETTINGS - the only numbers and paths this file uses
# =============================================================================

CONFIGS = Path(__file__).resolve().parents[1] / "configs"   # works from any folder

STOPWORDS_FILE = CONFIGS / "bn_stopwords.txt"
PROTECTED_WORDS_FILE = CONFIGS / "bn_protected_words.txt"
SUFFIXES_FILE = CONFIGS / "bn_suffixes.txt"

# The stemmer only cuts an ending if at least this many characters are left.
# "Characters" are Unicode code points, so a vowel sign such as া counts as one.
#   3 keeps "ধারা" (stream) whole instead of cutting it to "ধা".
MIN_STEM_LENGTH = 3

# The stemmer only cuts an ending if what is left is a REAL word: one that appears
# at least this many times in the TRAIN split text. Without this, "শব্দের" (of the
# word) would be cut at "দের" into the non-word "শব্". Same threshold as Skip-gram.
KNOWN_WORD_MIN_COUNT = 2

# A word ending in one of these is never stemmed, because the ending is a negation:
#   "করেননি" (did not do) must never become "করেন" (does).
NEGATION_ENDINGS = ("নি", "না")

# The preprocessing variants compared in the M10 experiment (guide §6).
VARIANTS = {
    "V0": "split on spaces only, no cleaning (lower bound)",
    "V1": "clean + tokenize (THE DEFAULT)",
    "V2": "V1 + remove stop words (protected words kept)",
    "V3": "V1 + stem (protected words kept)",
    "V4": "V1 + remove stop words + stem (protected words kept)",
    "V2-demo": "V1 + remove stop words with NO protection - shows the damage",
}


# =============================================================================
# PART 1 - CLEANING (Lab 1: regular expressions)
# Four small steps, always in this order. clean() runs all four.
# =============================================================================

def normalize_unicode(text: str) -> str:
    """Step 1: give every letter one single computer spelling (Unicode NFC).

    The letter য় can be stored as ONE code point (U+09DF) or as TWO (য + ়).
    They look identical on screen but compare as different strings, so a
    stop word typed one way would never match text typed the other way.
    """
    return unicodedata.normalize("NFC", text)


# A Wikipedia footnote mark: "[" + digits (English or Bengali) + "]".
# "(?!\{)" means "NOT followed by {". That keeps LaTeX maths such as
# \sqrt[7]{n^3} (the 7th root) - its [7] is always followed by "{".
CITATION_MARK = re.compile(r"\[[0-9০-৯]+\](?!\{)")


def remove_citation_marks(text: str) -> str:
    """Step 2: delete leftover Wikipedia footnote marks such as [1] or [২][৩].

    About 1 passage in 7 still carries them. They are formatting, not content.
    """
    return CITATION_MARK.sub("", text)


BENGALI_TO_ENGLISH_DIGITS = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")


def unify_digits(text: str) -> str:
    """Step 3: write every digit the same way: "১৯৭১" -> "1971".

    The number's VALUE never changes, so numeric errors ("১৭০৪" vs "১৭৫২")
    stay visible - but "১৯৭১" and "1971" now count as the same word.
    """
    return text.translate(BENGALI_TO_ENGLISH_DIGITS)


def collapse_whitespace(text: str) -> str:
    """Step 4: turn every run of spaces, tabs and newlines into one space."""
    return re.sub(r"\s+", " ", text).strip()


def clean(text: str | None) -> str:
    """Run the four cleaning steps in order.

    It never deletes Bengali letters, digits, punctuation or negation words.

    >>> clean("১৯৭১ সালে[১] স্বাধীন   হয়।")
    '1971 সালে স্বাধীন হয়।'
    """
    if not text:
        return ""
    text = normalize_unicode(text)
    text = remove_citation_marks(text)
    text = unify_digits(text)
    text = collapse_whitespace(text)
    return text


# =============================================================================
# PART 2 - TOKENIZING (Lab 1: tokenization)
# Cut cleaned text into a list of words, numbers and punctuation marks.
# =============================================================================

# One token is the FIRST of these that matches, tried left to right:
TOKEN_PATTERN = re.compile(
    r"[\u0980-\u09FF\u200C\u200D]+"   # a Bengali word (the two invisible zero-width joiners are kept)
    r"|[0-9]+(?:[.,][0-9]+)*"         # a number, with decimals or commas: 3.14, 1,000
    r"|[A-Za-z\u00C0-\u024F]+"        # a Latin-script word: DNA, École
    r"|[^\s]"                         # anything else, one character: । , ? $ ( -
)


def tokenize(text: str) -> list[str]:
    """Split CLEANED text into tokens. Nothing is lost.

    Joining the tokens back together gives the text without its spaces.
    The Bengali full stop (দাঁড়ি "।") becomes its own token.

    >>> tokenize("1971 সালে স্বাধীন হয়।")
    ['1971', 'সালে', 'স্বাধীন', 'হয়', '।']
    """
    return TOKEN_PATTERN.findall(text)


# =============================================================================
# PART 3 - WORD LISTS (read from configs/)
# Read once, then remembered (lru_cache), so they are not re-read for every sentence.
# =============================================================================

def read_word_list(path: Path) -> list[str]:
    """Read one entry per line. Skip blank lines and lines starting with "#"."""
    entries = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            entries.append(normalize_unicode(line))
    if not entries:
        raise ValueError(f"{path} contains no entries")
    return entries


@lru_cache(maxsize=None)
def stopwords() -> frozenset[str]:
    """The published Bengali stop-word list (configs/bn_stopwords.txt)."""
    return frozenset(read_word_list(STOPWORDS_FILE))


@lru_cache(maxsize=None)
def protected_words() -> frozenset[str]:
    """Negation and number words that are never removed and never stemmed."""
    return frozenset(read_word_list(PROTECTED_WORDS_FILE))


@lru_cache(maxsize=None)
def suffixes() -> tuple[str, ...]:
    """Stemmer endings, LONGEST FIRST, so "গুলোকে" is tried before "কে"."""
    return tuple(sorted(read_word_list(SUFFIXES_FILE), key=len, reverse=True))


@lru_cache(maxsize=None)
def known_words() -> frozenset[str]:
    """Words seen at least KNOWN_WORD_MIN_COUNT times in the TRAIN split (after V1).

    Only train is used, so no word from dev or test can shape how text is stemmed.
    """
    from splits import load_split      # imported here: only the stemmer needs data

    counts: collections.Counter = collections.Counter()
    for record in load_split("train"):
        for field in ("context", "question", "candidate_answer"):
            counts.update(tokenize(clean(record[field])))
    return frozenset(word for word, n in counts.items() if n >= KNOWN_WORD_MIN_COUNT)


# =============================================================================
# PART 4 - STOP-WORD REMOVAL (Lab 1) - used only by the M10 experiment
# =============================================================================

def remove_stopwords(tokens: list[str], keep_protected: bool = True) -> list[str]:
    """Drop tokens found in the stop-word list.

    keep_protected=True  (normal):  negation and number words always stay.
    keep_protected=False (V2-demo): the published list is used exactly as it is,
                                    to measure how much damage that does.
    """
    stop = stopwords()
    keep = protected_words() if keep_protected else frozenset()
    return [token for token in tokens if token not in stop or token in keep]


# =============================================================================
# PART 5 - STEMMING (Lab 1: Porter-style rules) - used only by the M10 experiment
# =============================================================================

def is_bengali_word(token: str) -> bool:
    """True if the token starts with a Bengali character (not a number or symbol)."""
    return "\u0980" <= token[0] <= "\u09FF"


def stem_word(word: str, known: frozenset[str] | None = None) -> str:
    """Cut ONE noun ending off a Bengali word, following these rules in order:

    1. Protected words (negation, numbers) are returned unchanged.
    2. Words ending in a negation ("নি", "না") are returned unchanged.
    3. Numbers, English words and punctuation are returned unchanged.
    4. Try endings longest first. Cut the first one where what is left
         a) has at least MIN_STEM_LENGTH characters, and
         b) is a known word (seen in the train text).
       If no ending passes both tests, return the word unchanged.

    `known` is the list of real words; leave it empty to use the train vocabulary.

    >>> stem_word("শব্দের")      # "দের" would leave the non-word "শব্", so "ের" is cut
    'শব্দ'
    """
    if word in protected_words():
        return word
    if word.endswith(NEGATION_ENDINGS):
        return word
    if not is_bengali_word(word):
        return word
    if known is None:
        known = known_words()
    for ending in suffixes():
        if not word.endswith(ending):
            continue
        stem_candidate = word[: -len(ending)]
        if len(stem_candidate) >= MIN_STEM_LENGTH and stem_candidate in known:
            return stem_candidate
    return word


def stem(tokens: list[str]) -> list[str]:
    """Stem every token. The list keeps the same length and order."""
    return [stem_word(token) for token in tokens]


# =============================================================================
# PART 6 - ONE ENTRY POINT FOR ALL MODELS
# =============================================================================

def preprocess(text: str | None, variant: str = "V1") -> list[str]:
    """Turn raw text into the tokens a model receives. See VARIANTS at the top.

    Models should call only this function, so changing a variant here changes
    it for every model at once.
    """
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}; choose from {list(VARIANTS)}")

    if variant == "V0":
        return (text or "").split()

    tokens = tokenize(clean(text))
    if variant in ("V2", "V4"):
        tokens = remove_stopwords(tokens, keep_protected=True)
    if variant == "V2-demo":
        tokens = remove_stopwords(tokens, keep_protected=False)
    if variant in ("V3", "V4"):
        tokens = stem(tokens)
    return tokens


# =============================================================================
# COMMAND LINE - show the steps, or run the safety checks
# =============================================================================

DEMO_SENTENCES = [
    "বাংলাদেশের সংবিধান ১৯৭২ সালের ৪ নভেম্বর গৃহীত হয়।[1][২]",   # citation marks, Bengali digits
    "সব বিভাগ আবার সাতটি অঞ্চলে বিভক্ত নয়।",                    # negation must survive
    "কলেজের ছাত্ররা একটি বই পড়েনি।",                              # stemming, number word, negation ending
    "$f(n) = \\sqrt[7]{n^3}$ হলে f(128) কত?",                     # LaTeX [7] must survive
    "  কান্তনগর মন্দিরের নির্মাণ\n১৭০৪ সালে   শুরু হয়। ",           # messy whitespace
]


def show_steps(text: str) -> None:
    """Print what every cleaning step and every variant does to one text."""
    print("=" * 72)
    print(f"INPUT              : {text!r}")
    step = normalize_unicode(text)
    print(f"1 unicode NFC      : {step!r}")
    step = remove_citation_marks(step)
    print(f"2 citation marks   : {step!r}")
    step = unify_digits(step)
    print(f"3 unify digits     : {step!r}")
    step = collapse_whitespace(step)
    print(f"4 whitespace       : {step!r}")
    print(f"tokens             : {tokenize(step)}")
    print("-" * 72)
    for name, description in VARIANTS.items():
        print(f"{name:<8} {preprocess(text, name)}")
        print(f"{'':<8} ({description})")


def run_check() -> int:
    """Check the rules on every train + dev record. Returns 0 if all pass, 1 if not.

    The test split is not read here - it stays untouched until the final evaluation.
    """
    from splits import load_split      # imported here so text_bn has no data dependency

    records = load_split("train") + load_split("dev")
    texts = [r[field] for r in records for field in ("context", "question", "candidate_answer")
             if r[field]]

    failures: collections.Counter = collections.Counter()
    examples: dict[str, str] = {}
    stats: collections.Counter = collections.Counter()
    vocab = {name: set() for name in VARIANTS}
    token_totals: collections.Counter = collections.Counter()
    stem_changes: collections.Counter = collections.Counter()
    demo_removed: collections.Counter = collections.Counter()
    stopword_removed: collections.Counter = collections.Counter()
    bengali = re.compile(r"[\u0980-\u09E5\u09F0-\u09FF]")    # Bengali block minus its digits

    def fail(rule: str, text: str) -> None:
        failures[rule] += 1
        examples.setdefault(rule, text[:120])

    protected = protected_words()
    for original in texts:
        nfc = normalize_unicode(original)
        cleaned = clean(original)
        stats["citation marks removed"] += len(CITATION_MARK.findall(nfc))
        stats["texts changed by NFC"] += nfc != original
        stats["Bengali digits converted"] += len(re.findall("[০-৯]", original))

        # Rule 1: cleaning twice gives the same result as cleaning once.
        if clean(cleaned) != cleaned:
            fail("clean() is not stable when run twice", original)

        # Rule 2: every digit survives, in the same order (only its script changes).
        expected_digits = re.findall("[0-9]", unify_digits(remove_citation_marks(nfc)))
        if re.findall("[0-9]", cleaned) != expected_digits:
            fail("a digit was lost or changed", original)

        # Rule 3: every Bengali letter survives.
        if len(bengali.findall(remove_citation_marks(nfc))) != len(bengali.findall(cleaned)):
            fail("a Bengali letter was lost", original)

        # Rule 4: LaTeX roots such as \sqrt[7]{...} are kept.
        if original.count("sqrt[") != cleaned.count("sqrt["):
            fail("a LaTeX root lost its [n]", original)

        # Rule 5: tokenizing loses nothing but spaces.
        tokens = tokenize(cleaned)
        if "".join(tokens) != cleaned.replace(" ", ""):
            fail("tokenize() lost characters", original)

        # Rule 6: protected words survive every normal variant.
        #   V2 removes words: every protected word in V1 must still be there.
        #   V3 = stem(V1) and V4 = stem(V2) keep word positions: a protected word must
        #   come out unchanged. (Stemming may CREATE a protected word - "চারটি" -> "চার"
        #   is correct - so we compare word by word, not by counting.)
        outputs = {name: preprocess(original, name) for name in VARIANTS}
        protected_in_v1 = collections.Counter(t for t in outputs["V1"] if t in protected)
        if collections.Counter(t for t in outputs["V2"] if t in protected) != protected_in_v1:
            fail("a protected word was removed or stemmed in V2", original)
        for name, source in (("V3", "V1"), ("V4", "V2")):
            if any(before in protected and after != before
                   for before, after in zip(outputs[source], outputs[name])):
                fail(f"a protected word was removed or stemmed in {name}", original)
        demo_removed.update(protected_in_v1 - collections.Counter(
            t for t in outputs["V2-demo"] if t in protected))
        stopword_removed.update(collections.Counter(outputs["V1"]) -
                                collections.Counter(outputs["V2"]))

        # Rule 7: words ending in a negation are never stemmed.
        for before, after in zip(outputs["V1"], outputs["V3"]):
            if before.endswith(NEGATION_ENDINGS) and before != after:
                fail("a negation ending was stemmed", original)
            elif before != after:
                stem_changes[(before, after)] += 1

        for name, out in outputs.items():
            vocab[name].update(out)
            token_totals[name] += len(out)

    # ---- report -------------------------------------------------------------
    print("=" * 72)
    print(f"TEXT_BN CHECK - {len(records):,} train+dev records, {len(texts):,} non-empty texts")
    print("=" * 72)
    for key, value in stats.items():
        print(f"  {key:<32}{value:>10,}")
    print(f"\n  {'variant':<9}{'tokens':>12}{'vocabulary':>13}   meaning")
    for name, description in VARIANTS.items():
        print(f"  {name:<9}{token_totals[name]:>12,}{len(vocab[name]):>13,}   {description}")
    v1 = token_totals["V1"]
    print(f"\n  stemming changed {sum(stem_changes.values()):,} of {v1:,} tokens "
          f"({sum(stem_changes.values()) / v1:.1%}); most common:")
    for (before, after), count in stem_changes.most_common(10):
        print(f"      {before} -> {after}  ({count:,})")
    print(f"\n  V2 removed {sum(stopword_removed.values()):,} stop words; most common "
          f"(look for words that carry meaning): "
          f"{', '.join(f'{w} ({n:,})' for w, n in stopword_removed.most_common(12))}")
    print(f"\n  V2-demo removed {sum(demo_removed.values()):,} protected words that V2 keeps; "
          f"most common: {', '.join(f'{w} ({n:,})' for w, n in demo_removed.most_common(8))}")

    print("\n  RULES")
    rules = ["clean() is not stable when run twice", "a digit was lost or changed",
             "a Bengali letter was lost", "a LaTeX root lost its [n]", "tokenize() lost characters",
             "a protected word was removed or stemmed in V2", "a protected word was removed or stemmed in V3",
             "a protected word was removed or stemmed in V4", "a negation ending was stemmed"]
    for rule in rules:
        mark = "PASS" if failures[rule] == 0 else f"FAIL x{failures[rule]}"
        print(f"    [{mark}] never: {rule}")
        if rule in examples:
            print(f"           e.g. {examples[rule]!r}")
    print("=" * 72)
    return 0 if not failures else 1


def main() -> int:
    parser = argparse.ArgumentParser(description="Bengali text tools (NLP Lab 1, M10).")
    parser.add_argument("text", nargs="?", help="a sentence to show step by step")
    parser.add_argument("--demo", action="store_true", help="show every step on built-in examples")
    parser.add_argument("--check", action="store_true", help="run the safety checks on train + dev")
    args = parser.parse_args()

    if args.check:
        return run_check()
    if args.demo:
        for sentence in DEMO_SENTENCES:
            show_steps(sentence)
        return 0
    if args.text:
        show_steps(args.text)
        return 0
    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
