"""Hand-made features: a character language model and similarity scores (NLP Labs 1-3 -> M11, M12).

A bag-of-words model can see WHICH words an answer uses, but not whether those words
match the passage in front of it. These functions add exactly that, using only the
record's own question and passage - never any other record (PRD 5.3a).

    from features import extract_features, passage_lm_score, best_match_distance

    extract_features(record)        -> {"exact_in_passage": 1.0, "edit_passage": 0.0, ...}
    best_match_distance("১৭০৪", passage)   -> how many letters must change to find it

Run from the repository root:

    python src/features.py --demo        # show the features for a few real dev pairs
    python src/features.py --check       # sanity rules + how much signal each feature carries
    python src/features.py --baseline    # tune and score the V6 fuzzy string matcher
    python src/features.py --baseline --log    # ...and append the result to the experiment log

WHAT COMES FROM WHICH LAB
  Lab 1  edit distance (the dynamic-programming table)     -> edit_passage, edit_question, V6
  Lab 2  n-gram language model (counts + Laplace smoothing) -> lm_passage, and the M11b classifier
  Lab 2  word overlap                                       -> exact_in_passage, token_overlap
  Lab 3  cosine similarity of word vectors                  -> added at M2, once Skip-gram exists

TWO THINGS THE DATA FORCED ON THIS FILE
  * Passages run to 3,127 characters while answers are 16 on average. Comparing the answer
    to every window of the passage separately would take billions of steps, so
    best_match_distance() uses the standard "free start" edit-distance table instead: one
    pass that finds the best-matching piece of the passage anywhere inside it.
  * 449 answers in train+dev are 3 characters or shorter ("3", "৮", "১০"). A short answer
    matches almost any passage by luck, so `answer_len` is kept as a feature and the
    reports below always show the short-answer group separately.
"""

from __future__ import annotations

import argparse
import collections
import csv
import math
import sys
from datetime import date
from pathlib import Path

import numpy as np

from sklearn.metrics import roc_auc_score

# The scoring lives in one place, so a change to it reaches every file at once.
# (evaluate.py imports THIS file only inside a function, so the two do not clash.)
from evaluate import macro_f1
from text_bn import clean, tokenize

# =============================================================================
# SETTINGS
# =============================================================================

# How many characters of history the language model looks at: 4 means "predict this
# character from the three before it".
#
# How much smoothing to add to every count. The lab uses Laplace (add 1). On this data
# add-1 is too blunt: with roughly 60 different characters, a character the passage HAS
# seen scores only twice a character it has never seen, so a copied answer and an
# invented one look nearly the same. A smaller number sharpens that contrast.
#
# Both were chosen by their AUC on the TRAIN split, hard subset (the number this project
# is actually judged on). Run `python src/features.py --tune-lm` to reproduce:
#
#       order   k     train hard AUC    dev hard AUC
#         3    1.0        0.606            0.640      <- the plain lab settings
#         3    0.1        0.605            0.641
#         4    1.0        0.603            0.640
#         4    0.1        0.611            0.657      <- chosen
#
# (AUC 0.50 would mean the feature is useless; 1.00 would mean perfect.)
NGRAM_ORDER = 4
SMOOTHING_K = 0.1

# The V6 fuzzy rule's cut-off: "call the answer correct if at most this share of its
# characters must change to find it somewhere in the passage". Tuned on the TRAIN hard
# subset by tune_fuzzy_threshold(records, subset="hard") - rerun `--baseline` after any
# change to the features and update this number. evaluate.py imports it from here rather
# than keeping its own copy, so the two can never drift apart.
FUZZY_THRESHOLD = 0.405

# Answers this short match a passage by luck, so reports separate them out.
SHORT_ANSWER_CHARS = 4

# What the passage features mean when a record HAS NO PASSAGE (no-context items).
# Each one means "no evidence of a match", which is why they are not all zero:
# a distance of 0 would wrongly mean "found it exactly".
NO_PASSAGE = {
    "exact_in_passage": 0.0,   # 1 = the answer appears word for word
    "edit_passage": 1.0,       # 0 = found exactly, 1 = nothing alike
    "token_overlap": 0.0,      # 1 = every word of the answer is in the passage
    "lm_passage": 0.0,         # 1 = the passage would very likely produce this answer
}

# The start-of-text marker the language model pads with, like <s> in the lab.
# \x02 is a control character, so it can never appear in real Bengali text.
START = "\x02"

LOG_FILE = Path(__file__).resolve().parents[1] / "results" / "experiment_log.csv"


# =============================================================================
# PART 1 - EDIT DISTANCE (Lab 1)
# =============================================================================

def edit_distance(a: str, b: str) -> int:
    """The Lab 1 dynamic-programming table: how many single-character edits turn a into b.

    An edit is an insert, a delete, or a replace. This is the plain version, kept
    because it is the one from the lab and the one the tests check everything else against.

    >>> edit_distance("kitten", "sitting")
    3
    """
    previous = list(range(len(b) + 1))
    for i, ch_a in enumerate(a, start=1):
        current = [i]
        for j, ch_b in enumerate(b, start=1):
            current.append(min(
                previous[j] + 1,                        # delete a character of a
                current[j - 1] + 1,                     # insert a character of b
                previous[j - 1] + (ch_a != ch_b),       # replace (free if they match)
            ))
        previous = current
    return previous[-1]


def best_match_distance(needle: str, haystack: str) -> int:
    """Fewest edits needed to find `needle` somewhere inside `haystack`.

    This is the same table as edit_distance, with one change: the first row is all
    zeros, which means "the match may start at any position". So it answers
    "what is the closest thing to this answer anywhere in this passage?" rather than
    "how different is the answer from the whole passage?".

    Everything is done with numpy rows because passages reach 3,127 characters.

    >>> best_match_distance("১৭০৪", "নির্মাণ ১৭০৪ সালে শুরু")   # present exactly
    0
    >>> best_match_distance("১৭০৫", "নির্মাণ ১৭০৪ সালে শুরু")   # one digit off
    1
    """
    if not needle:
        return 0
    if not haystack:
        return len(needle)

    # Unicode code points, so numpy can compare a whole passage at once.
    hay = np.array([ord(c) for c in haystack], dtype=np.int32)
    positions = np.arange(len(haystack) + 1, dtype=np.int32)

    previous = np.zeros(len(haystack) + 1, dtype=np.int32)   # free start: match may begin anywhere
    for i, ch in enumerate(needle, start=1):
        replace_or_keep = previous[:-1] + (hay != ord(ch))
        delete = previous[1:] + 1
        current = np.empty_like(previous)
        current[0] = i                                       # drop the first i answer characters
        current[1:] = np.minimum(replace_or_keep, delete)

        # The "insert" move needs cell j-1 of THIS row, which numpy cannot do in one step.
        # min over k <= j of (current[k] + (j - k)) is the same thing, and a running
        # minimum computes it in one pass.
        current = np.minimum.accumulate(current - positions) + positions
        previous = current

    return int(previous.min())    # free end: the match may stop anywhere


def normalised_best_match(needle: str, haystack: str) -> float:
    """best_match_distance as a 0-to-1 score: 0 = found exactly, 1 = nothing alike.

    Divided by the length of the needle, because that is the most edits the answer
    can ever need (delete every character of it).
    """
    if not needle:
        return 1.0
    return best_match_distance(needle, haystack) / len(needle)


# =============================================================================
# PART 2 - CHARACTER N-GRAM LANGUAGE MODEL (Lab 2 counts + Lab 3 smoothing)
# =============================================================================

class CharLanguageModel:
    """Predicts the next character from the few before it - the Lab 2 model, per character.

    Why characters and not words: our answers are 16 characters long on average, so a
    word-level model would see two or three words and have nothing to count.

    Why Laplace smoothing (Lab 3): plain counts give probability 0 to anything unseen,
    and one zero makes the whole answer score minus infinity. Adding 1 to every count
    keeps unseen characters merely unlikely instead of impossible.
    """

    def __init__(self, order: int = NGRAM_ORDER, smoothing: float = SMOOTHING_K) -> None:
        self.order = order
        self.smoothing = smoothing
        self.counts: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
        self.history_totals: collections.Counter = collections.Counter()
        self.alphabet: set[str] = set()

    def fit(self, texts: list[str]) -> CharLanguageModel:
        """Count every (history -> next character) pair in these texts."""
        for text in texts:
            padded = START * (self.order - 1) + text
            self.alphabet.update(text)
            for i in range(self.order - 1, len(padded)):
                history = padded[i - self.order + 1:i]
                self.counts[history][padded[i]] += 1
                self.history_totals[history] += 1
        return self

    def observe_alphabet(self, text: str) -> None:
        """Let the model know about characters it may be asked to score.

        Without this, a character the training text never used would be treated as
        impossible rather than rare.
        """
        self.alphabet.update(text)

    def char_probability(self, history: str, char: str) -> float:
        """Smoothed probability - the lab's formula with the added amount left adjustable.

            (times we saw this character after this history  +  k)
            -------------------------------------------------------
            (times we saw this history  +  k x how many characters exist)

        k = 1 is the lab's Laplace smoothing. This project uses k = SMOOTHING_K; the
        comment at the top of the file shows the measurements behind that choice.
        """
        vocabulary = max(len(self.alphabet), 1)
        return ((self.counts[history][char] + self.smoothing)
                / (self.history_totals[history] + self.smoothing * vocabulary))

    def average_log_probability(self, text: str) -> float:
        """Mean log probability per character.

        The mean (not the sum) so that a long answer is not punished for being long.
        """
        if not text:
            return 0.0
        padded = START * (self.order - 1) + text
        total = 0.0
        for i in range(self.order - 1, len(padded)):
            history = padded[i - self.order + 1:i]
            total += math.log(self.char_probability(history, padded[i]))
        return total / len(text)

    def typicality(self, text: str) -> float:
        """The score as a 0-to-1 number: the average chance of each character.

        Easier to read than a log probability, and 0 can then mean "no passage at all".
        """
        if not text:
            return 0.0
        return math.exp(self.average_log_probability(text))


def passage_lm_score(record: dict, order: int = NGRAM_ORDER,
                     smoothing: float = SMOOTHING_K) -> float:
    """M11(a): how much does this answer sound like it came from ITS OWN passage?

    A language model is built from that one passage and nothing else, then asked how
    likely the answer's characters are. Unlike exact string matching, this still gives
    credit when a single letter differs - which is most of the hard cases.

    Returns 0.0 for records with no passage.
    """
    passage, answer = clean(record["context"]), clean(record["candidate_answer"])
    if not passage or not answer:
        return NO_PASSAGE["lm_passage"]
    model = CharLanguageModel(order, smoothing).fit([passage])
    model.observe_alphabet(answer)     # characters the passage never used are rare, not impossible
    return model.typicality(answer)


class AnswerStyleClassifier:
    """M11(b): one language model for correct answers, one for wrong ones.

    It reads ONLY the answer - no question, no passage. So it cannot possibly check
    whether an answer is true; it can only notice that wrong answers are *written*
    differently. That makes it a warning light: guide 9.2 says to report it next to
    the answer-only probe (0.542). A high score here means the wrong answers carry a
    fingerprint from whatever wrote them, which is a finding about the data.
    """

    def __init__(self, order: int = NGRAM_ORDER, smoothing: float = SMOOTHING_K) -> None:
        self.order = order
        self.smoothing = smoothing
        self.models: dict[int, CharLanguageModel] = {}

    def fit(self, records: list[dict]) -> AnswerStyleClassifier:
        by_label: dict[int, list[str]] = {0: [], 1: []}
        for record in records:
            by_label[record["label"]].append(clean(record["candidate_answer"]))
        shared_alphabet = {ch for texts in by_label.values() for text in texts for ch in text}
        for label, texts in by_label.items():
            model = CharLanguageModel(self.order, self.smoothing).fit(texts)
            model.alphabet |= shared_alphabet      # both models judge on the same alphabet
            self.models[label] = model
        return self

    def score_gap(self, record: dict) -> float:
        """log P(answer | correct-answer model) - log P(answer | wrong-answer model).

        Positive means it reads more like a correct answer.
        """
        answer = clean(record["candidate_answer"])
        return (self.models[1].average_log_probability(answer)
                - self.models[0].average_log_probability(answer))

    def predict(self, record: dict) -> int:
        return 1 if self.score_gap(record) > 0 else 0


# =============================================================================
# PART 3 - THE FEATURE SET (M12)
# =============================================================================

def is_word_token(token: str) -> bool:
    """True for real words and numbers; False for punctuation such as "।" or "$"."""
    return any(ch.isalnum() for ch in token)


def extract_features(record: dict, order: int = NGRAM_ORDER) -> dict[str, float]:
    """All of a record's similarity features, in one dictionary.

    Every value compares the record with ITSELF - its own question and its own passage.
    Nothing here looks at another record, so this is not retrieval (PRD 5.3a).

    Feature directions (they are not all the same, on purpose):
        exact_in_passage  1 = found word for word
        edit_passage      0 = found exactly,      1 = nothing alike
        edit_question     0 = repeats the question, 1 = nothing alike
        token_overlap     1 = every answer word appears in the passage
        lm_passage        1 = the passage would readily produce this answer
        has_context       1 = a passage exists
        answer_len        characters in the answer
        digit_share       share of the answer that is digits
    """
    answer = clean(record["candidate_answer"])
    question = clean(record["question"])
    passage = clean(record["context"])

    features: dict[str, float] = {
        "has_context": 1.0 if passage else 0.0,
        "answer_len": float(len(answer)),
        "digit_share": sum(ch.isdigit() for ch in answer) / len(answer) if answer else 0.0,
        "edit_question": normalised_best_match(answer, question) if question else 1.0,
    }

    if not passage:
        features.update(NO_PASSAGE)
        return features

    answer_tokens = [t for t in tokenize(answer) if is_word_token(t)]
    passage_tokens = set(tokenize(passage))
    trimmed = answer.rstrip("।. ")      # a trailing full stop should not break an exact match

    # Both use the trimmed answer, so a trailing "।" cannot make the two disagree.
    # (It did: 198 train answers such as "আরবি।" are in the passage without the full stop.)
    features["exact_in_passage"] = 1.0 if trimmed and trimmed in passage else 0.0
    features["edit_passage"] = normalised_best_match(trimmed, passage)
    features["token_overlap"] = (sum(t in passage_tokens for t in answer_tokens) / len(answer_tokens)
                                 if answer_tokens else 0.0)
    features["lm_passage"] = passage_lm_score(record, order)
    return features


FEATURE_NAMES = ["exact_in_passage", "edit_passage", "edit_question", "token_overlap",
                 "lm_passage", "has_context", "answer_len", "digit_share"]


def feature_matrix(records: list[dict], order: int = NGRAM_ORDER) -> np.ndarray:
    """Features for many records, as rows in the fixed FEATURE_NAMES order."""
    return np.array([[extract_features(r, order)[name] for name in FEATURE_NAMES]
                     for r in records], dtype=np.float64)


# =============================================================================
# PART 4 - THE V6 FUZZY STRING-MATCHER BASELINE
# =============================================================================

def tune_fuzzy_threshold(records: list[dict], subset: str = "all") -> tuple[float, float]:
    """Pick the edit_passage cut-off that scores best ON TRAIN. Returns (threshold, its score).

    V6: "call it correct if the answer NEARLY appears in the passage". The exact matcher
    fails the moment one letter changes; this rule forgives that.

    subset="all"  tunes on every has-context train record.
    subset="hard" tunes on the hard ones only - and the two answers are different,
                  because loosening the rule helps on hard records and hurts everywhere
                  else. See run_baseline() for the numbers.
    """
    grounded = [r for r in records if r["condition"] == "has_context" and r["context"]]
    if subset == "hard":
        grounded = [r for r in grounded if r.get("difficulty") == "hard"]
    distances = [extract_features(r)["edit_passage"] for r in grounded]
    truth = [r["label"] for r in grounded]

    best_threshold, best_score = 0.0, -1.0
    for threshold in np.arange(0.0, 1.005, 0.005):
        predicted = [1 if d <= threshold else 0 for d in distances]
        score = macro_f1(truth, predicted)
        if score > best_score:
            best_threshold, best_score = round(float(threshold), 3), score
    return best_threshold, best_score


def score_fuzzy(records: list[dict], threshold: float) -> dict[str, tuple[float, int]]:
    """Score the V6 rule on a split, split out by difficulty. Returns {slice: (macro-F1, n)}."""
    grounded = [r for r in records if r["condition"] == "has_context" and r["context"]]
    distances = [extract_features(r)["edit_passage"] for r in grounded]
    truth = [r["label"] for r in grounded]
    predicted = [1 if d <= threshold else 0 for d in distances]

    results = {"all": (macro_f1(truth, predicted), len(truth))}
    for level in ("easy", "hard"):
        rows = [i for i, r in enumerate(grounded) if r.get("difficulty") == level]
        if rows:
            results[level] = (macro_f1([truth[i] for i in rows], [predicted[i] for i in rows]),
                              len(rows))
    return results


def score_exact(records: list[dict]) -> dict[str, tuple[float, int]]:
    """The same, for the plain exact-substring rule, so the two can be compared fairly."""
    grounded = [r for r in records if r["condition"] == "has_context" and r["context"]]
    truth = [r["label"] for r in grounded]
    predicted = [int(extract_features(r)["exact_in_passage"]) for r in grounded]

    results = {"all": (macro_f1(truth, predicted), len(truth))}
    for level in ("easy", "hard"):
        rows = [i for i, r in enumerate(grounded) if r.get("difficulty") == level]
        if rows:
            results[level] = (macro_f1([truth[i] for i in rows], [predicted[i] for i in rows]),
                              len(rows))
    return results


# =============================================================================
# COMMAND LINE
# =============================================================================

def run_demo() -> int:
    """Show the features for a few real dev pairs, correct answer beside wrong answer."""
    from splits import load_split

    dev = load_split("dev")
    pairs: dict[str, list[dict]] = collections.defaultdict(list)
    for record in dev:
        pairs[record["pair_id"]].append(record)

    wanted = [("has_context", "easy"), ("has_context", "hard"), ("no_context", "hard")]
    shown = 0
    for condition, difficulty in wanted:
        for pair in pairs.values():
            if (len(pair) == 2 and pair[0]["condition"] == condition
                    and pair[0]["difficulty"] == difficulty):
                correct = next(r for r in pair if r["label"] == 1)
                wrong = next(r for r in pair if r["label"] == 0)
                print("=" * 78)
                print(f"{condition} / {difficulty} / {correct['subject']}")
                if correct["context"]:
                    print(f"  passage : {clean(correct['context'])[:110]}...")
                print(f"  question: {clean(correct['question'])[:110]}")
                print(f"  CORRECT : {clean(correct['candidate_answer'])[:70]}")
                print(f"  WRONG   : {clean(wrong['candidate_answer'])[:70]}")
                good, bad = extract_features(correct), extract_features(wrong)
                print(f"\n  {'feature':<18}{'correct':>10}{'wrong':>10}   what a gap here would mean")
                meanings = {
                    "exact_in_passage": "the words are in the passage",
                    "edit_passage": "lower = closer to the passage",
                    "edit_question": "lower = closer to the question",
                    "token_overlap": "share of answer words in the passage",
                    "lm_passage": "higher = reads like the passage",
                    "answer_len": "characters",
                    "digit_share": "share of digits",
                }
                for name, meaning in meanings.items():
                    print(f"  {name:<18}{good[name]:>10.3f}{bad[name]:>10.3f}   {meaning}")
                shown += 1
                break
    return 0 if shown else 1


def run_check() -> int:
    """Sanity rules, plus how much each feature actually separates right from wrong."""
    from splits import load_split

    train, dev = load_split("train"), load_split("dev")
    failures: list[str] = []

    # ---- rules the code must obey -------------------------------------------
    if best_match_distance("১৭০৪", "নির্মাণ ১৭০৪ সালে") != 0:
        failures.append("an exact match should cost 0 edits")
    if best_match_distance("১৭০৫", "নির্মাণ ১৭০৪ সালে") != 1:
        failures.append("a one-digit difference should cost 1 edit")
    if edit_distance("kitten", "sitting") != 3:
        failures.append("the Lab 1 example (kitten/sitting = 3) is wrong")

    # The fast version must give the same answer as checking every possible piece of the
    # passage with the plain Lab 1 table. Short strings only - the slow way is O(n squared)
    # pieces, each needing its own table.
    rng = np.random.default_rng(42)
    grounded = [r for r in train if r["context"]]
    for record in [grounded[i] for i in rng.integers(0, len(grounded), 25)]:
        answer, passage = clean(record["candidate_answer"])[:8], clean(record["context"])[:60]
        if not answer:
            continue
        slow = min(edit_distance(answer, passage[start:end])
                   for start in range(len(passage) + 1)
                   for end in range(start, len(passage) + 1))
        if best_match_distance(answer, passage) != slow:
            failures.append(f"fast and slow edit distance disagree on {answer!r}")
            break

    # the plain table must agree with a trusted library (guide 7.1b asks for this check)
    try:
        from rapidfuzz.distance import Levenshtein
        words = list({clean(r["candidate_answer"])[:25] for r in train[:1200]})
        mismatches = sum(edit_distance(a, b) != Levenshtein.distance(a, b)
                         for a, b in zip(words[:500], words[1:501]))
        library_note = f"checked against rapidfuzz on 500 pairs, {mismatches} disagreements"
        if mismatches:
            failures.append("edit_distance disagrees with rapidfuzz")
    except ImportError:
        library_note = "rapidfuzz not installed, cross-check skipped"

    # no-context records must not look like perfect matches
    no_context = next(r for r in dev if r["condition"] == "no_context")
    if extract_features(no_context)["edit_passage"] != 1.0:
        failures.append("a record with no passage must score 1.0 (nothing alike), not 0.0")

    # ---- how much signal is in each feature ---------------------------------
    print("=" * 78)
    print(f"FEATURES CHECK - train {len(train):,} + dev {len(dev):,} records")
    print("=" * 78)
    print(f"  {library_note}")

    grounded_dev = [r for r in dev if r["context"]]
    hard = [r for r in grounded_dev if r["difficulty"] == "hard"]
    print("\n  How well each feature separates correct from wrong answers, on dev.")
    print("  AUC is the chance it ranks a random correct answer above a random wrong one:")
    print("  0.50 = useless, 1.00 = perfect. Below 0.50 just means the feature points downward.")
    for title, subset in (("all dev", dev), ("has-context", grounded_dev),
                          ("has-context HARD", hard)):
        features = [extract_features(r) for r in subset]
        labels = [r["label"] for r in subset]
        print(f"\n  {title} ({len(subset):,} records)")
        print(f"  {'feature':<18}{'correct':>10}{'wrong':>10}{'AUC':>8}")
        for name in FEATURE_NAMES:
            values = [f[name] for f in features]
            good = sum(v for v, y in zip(values, labels) if y == 1) / max(sum(labels), 1)
            bad = sum(v for v, y in zip(values, labels) if y == 0) / max(len(labels) - sum(labels), 1)
            auc = roc_auc_score(labels, values) if len(set(values)) > 1 else 0.5
            print(f"  {name:<18}{good:>10.3f}{bad:>10.3f}{auc:>8.3f}")

    # short answers match by luck - show how many and how they behave
    short = [r for r in grounded_dev if len(clean(r["candidate_answer"])) < SHORT_ANSWER_CHARS]
    if short:
        hits = sum(extract_features(r)["exact_in_passage"] for r in short)
        print(f"\n  answers under {SHORT_ANSWER_CHARS} characters: {len(short):,} of "
              f"{len(grounded_dev):,} has-context dev records; {hits / len(short):.0%} of them "
              f"'appear in the passage' - mostly by luck, which is why answer_len is a feature")

    # ---- the language model, at three orders --------------------------------
    print(f"\n  M11(b) answer-style classifier on dev - compare with the answer-only probe (0.542)")
    for order in (2, 3, 4):
        classifier = AnswerStyleClassifier(order).fit(train)
        predicted = [classifier.predict(r) for r in dev]
        score = macro_f1([r["label"] for r in dev], predicted)
        flag = "  <- much higher than the probe: wrong answers have a writing style" if score > 0.62 else ""
        print(f"    order {order}: macro-F1 {score:.3f}{flag}")

    print("\n  RULES")
    for rule in ["an exact match should cost 0 edits",
                 "a one-digit difference should cost 1 edit",
                 "the Lab 1 example (kitten/sitting = 3) is wrong",
                 "fast and slow edit distance agree",
                 "edit_distance agrees with rapidfuzz",
                 "a record with no passage scores 1.0, not 0.0"]:
        broken = any(rule.split(" agree")[0] in f or rule in f for f in failures)
        print(f"    [{'FAIL' if broken else 'PASS'}] {rule}")
    print("=" * 78)
    return 1 if failures else 0


def run_tune_lm() -> int:
    """Reproduce the table at the top of this file: which history length and smoothing to use.

    Chosen on TRAIN (dev is only shown to confirm the choice was not a fluke), and judged
    on the HARD subset, because that is the number this project is actually measured on.
    """
    from splits import load_split

    train = [r for r in load_split("train") if r["context"]]
    dev = [r for r in load_split("dev") if r["context"]]

    print("=" * 78)
    print("TUNING THE PASSAGE LANGUAGE MODEL (M11a) - chosen on train, confirmed on dev")
    print("=" * 78)
    print(f"  {'order':>6}{'k':>7}{'train AUC':>11}{'train hard':>12}{'dev AUC':>10}{'dev hard':>10}")
    best = None
    for order in (2, 3, 4):
        for k in (1.0, 0.1, 0.05, 0.01):
            line = {}
            for name, rows in (("train", train), ("dev", dev)):
                values = [passage_lm_score(r, order, k) for r in rows]
                labels = [r["label"] for r in rows]
                hard_rows = [i for i, r in enumerate(rows) if r["difficulty"] == "hard"]
                line[name] = (
                    roc_auc_score(labels, values),
                    roc_auc_score([labels[i] for i in hard_rows], [values[i] for i in hard_rows]),
                )
            chosen = " <- in use" if (order, k) == (NGRAM_ORDER, SMOOTHING_K) else ""
            print(f"  {order:>6}{k:>7}{line['train'][0]:>11.3f}{line['train'][1]:>12.3f}"
                  f"{line['dev'][0]:>10.3f}{line['dev'][1]:>10.3f}{chosen}")
            if best is None or line["train"][1] > best[0]:
                best = (line["train"][1], order, k)
    print(f"\n  best by train hard AUC: order {best[1]}, k = {best[2]}")
    print(f"  currently set in this file: order {NGRAM_ORDER}, k = {SMOOTHING_K}")
    print("=" * 78)
    return 0


def run_baseline(write_log: bool) -> int:
    """Tune the V6 fuzzy rule on train, score it on dev, and compare with the exact rule."""
    from splits import load_split

    train, dev = load_split("train"), load_split("dev")
    threshold, train_score = tune_fuzzy_threshold(train)
    hard_threshold, hard_train_score = tune_fuzzy_threshold(train, subset="hard")

    exact = score_exact(dev)
    fuzzy = score_fuzzy(dev, threshold)
    fuzzy_hard = score_fuzzy(dev, hard_threshold)

    print("=" * 78)
    print("V6 - FUZZY STRING-MATCHER BASELINE")
    print("=" * 78)
    print("  The rule: call the answer correct if only a small share of its characters")
    print("  would have to change to find it somewhere in the passage.\n")
    print(f"  Cut-off tuned on ALL train has-context records : {threshold:.3f} "
          f"(train macro-F1 {train_score:.3f})")
    print(f"  Cut-off tuned on the HARD train records only   : {hard_threshold:.3f} "
          f"(train macro-F1 {hard_train_score:.3f})")
    print("\n  Why two: loosening the rule HURTS overall (easy records are already solved by")
    print("  exact matching, so every extra allowance just lets wrong answers through) but")
    print("  HELPS on hard records, where exact matching has nothing to work with.\n")

    print(f"  {'dev slice':<10}{'exact':>9}{'fuzzy':>9}{'fuzzy-hard':>12}{'n':>8}   "
          f"(fuzzy tuned on all / on hard)")
    for slice_name in ("all", "easy", "hard"):
        if slice_name in fuzzy:
            print(f"  {slice_name:<10}{exact[slice_name][0]:>9.3f}{fuzzy[slice_name][0]:>9.3f}"
                  f"{fuzzy_hard[slice_name][0]:>12.3f}{fuzzy[slice_name][1]:>8,}")
    print("\n  The bar a real model has to clear is the BEST of these on each row.")
    print("  On the hard subset that is the fuzzy rule, not the exact one.")
    print("=" * 78)

    if write_log:
        rows = list(csv.reader(LOG_FILE.open(encoding="utf-8")))
        existing = {r[0] for r in rows[1:] if r}
        new_rows = []
        plan = [("all", "baseline_001", fuzzy, threshold, "tuned on all train has-context"),
                ("easy", "baseline_002", fuzzy, threshold, "tuned on all train has-context"),
                ("hard", "baseline_003", fuzzy, threshold, "tuned on all train has-context"),
                ("hard", "baseline_004", fuzzy_hard, hard_threshold, "tuned on train HARD only")]
        for slice_name, run_id, table, cut, how in plan:
            if run_id in existing or slice_name not in table:
                continue
            score, n = table[slice_name]
            new_rows.append([run_id, date.today().isoformat(), f"fuzzy_substring_rule_{slice_name}",
                             "rule_based", "text_bn_V1", "42", "", "", "", "dev_usable",
                             f"{score:.3f}", "",
                             f"V6 fuzzy matcher, edit_passage <= {cut:.3f} ({how}); "
                             f"has-context {slice_name} (n={n}); exact rule scores "
                             f"{exact[slice_name][0]:.3f} on the same records"])
        if new_rows:
            with LOG_FILE.open("a", encoding="utf-8", newline="") as fh:
                csv.writer(fh).writerows(new_rows)
            print(f"  logged {len(new_rows)} row(s) to {LOG_FILE.name}")
        else:
            print("  nothing logged: those run_ids are already in the log")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Similarity features and the character LM (M11, M12).")
    parser.add_argument("--demo", action="store_true", help="show features for a few real dev pairs")
    parser.add_argument("--check", action="store_true", help="sanity rules + signal report")
    parser.add_argument("--baseline", action="store_true", help="tune and score the V6 fuzzy rule")
    parser.add_argument("--tune-lm", action="store_true", help="redo the language-model settings table")
    parser.add_argument("--log", action="store_true", help="with --baseline: append to the experiment log")
    args = parser.parse_args()

    if args.check:
        return run_check()
    if args.tune_lm:
        return run_tune_lm()
    if args.baseline:
        return run_baseline(args.log)
    if args.demo:
        return run_demo()
    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
