"""Tests for src/features.py - one test per rule, each with a plain-language name.

Run from the repository root:

    python -m pytest tests/test_features.py -v

These use tiny made-up strings, so they run in a second and do not need the corpus.
The checks that DO need the real data live in `python src/features.py --check`.
"""

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from features import (  # noqa: E402  (the import has to come after the path is set)
    FEATURE_NAMES,
    NO_PASSAGE,
    AnswerStyleClassifier,
    CharLanguageModel,
    best_match_distance,
    edit_distance,
    extract_features,
    macro_f1,
    normalised_best_match,
)


def record(answer: str, passage: str = "", question: str = "প্রশ্ন কী?", label: int = 1) -> dict:
    """A minimal record, shaped like the ones in data/splits/."""
    return {
        "candidate_answer": answer,
        "context": passage,
        "question": question,
        "label": label,
        "condition": "has_context" if passage else "no_context",
    }


# ----------------------------------------------------------- edit distance ---

def test_the_lab_1_example():
    assert edit_distance("kitten", "sitting") == 3


def test_identical_strings_need_no_edits():
    assert edit_distance("ঢাকা", "ঢাকা") == 0


def test_distance_to_an_empty_string_is_its_length():
    assert edit_distance("ঢাকা", "") == 4


def test_an_answer_inside_the_passage_costs_nothing():
    assert best_match_distance("১৭০৪", "নির্মাণ ১৭০৪ সালে শুরু হয়") == 0


def test_one_wrong_digit_costs_one_edit():
    assert best_match_distance("১৭০৫", "নির্মাণ ১৭০৪ সালে শুরু হয়") == 1


def test_nothing_in_common_costs_the_whole_answer():
    assert best_match_distance("abcd", "xyz") == 4


def test_empty_passage_costs_the_whole_answer():
    assert best_match_distance("ঢাকা", "") == 4


def test_fast_search_matches_checking_every_piece_by_hand():
    """The fast version must agree with the slow, obvious way on random small cases."""
    random.seed(42)
    for _ in range(200):
        needle = "".join(random.choice("abcd") for _ in range(random.randint(1, 5)))
        hay = "".join(random.choice("abcd") for _ in range(random.randint(0, 12)))
        slow = min(edit_distance(needle, hay[i:j])
                   for i in range(len(hay) + 1) for j in range(i, len(hay) + 1))
        assert best_match_distance(needle, hay) == slow, (needle, hay)


def test_normalised_distance_stays_between_zero_and_one():
    assert normalised_best_match("১৭০৪", "নির্মাণ ১৭০৪ সালে") == 0.0
    assert normalised_best_match("abcd", "xyz") == 1.0
    assert 0.0 < normalised_best_match("১৭০৫", "নির্মাণ ১৭০৪ সালে") < 1.0


# -------------------------------------------------------- language model ---

def test_probabilities_of_all_characters_add_up_to_one():
    model = CharLanguageModel(order=2).fit(["ababab"])
    total = sum(model.char_probability("a", ch) for ch in model.alphabet)
    assert abs(total - 1.0) < 1e-9


def test_a_character_the_model_never_saw_is_unlikely_but_not_impossible():
    model = CharLanguageModel(order=2).fit(["aaaa"])
    model.observe_alphabet("z")
    assert 0.0 < model.char_probability("a", "z") < model.char_probability("a", "a")


def test_text_like_the_training_text_scores_higher():
    model = CharLanguageModel(order=2).fit(["ঢাকা ঢাকা ঢাকা"])
    model.observe_alphabet("xyz")
    assert model.typicality("ঢাকা") > model.typicality("xyz")


def test_typicality_is_a_probability():
    model = CharLanguageModel(order=2).fit(["abcabc"])
    assert 0.0 < model.typicality("abc") <= 1.0
    assert model.typicality("") == 0.0


def test_the_answer_style_classifier_learns_an_obvious_difference():
    train = ([record("হ্যাঁ ঠিক আছে", label=1) for _ in range(20)]
             + [record("zzz qqq www", label=0) for _ in range(20)])
    classifier = AnswerStyleClassifier(order=2).fit(train)
    assert classifier.predict(record("হ্যাঁ ঠিক আছে")) == 1
    assert classifier.predict(record("zzz qqq www")) == 0


# --------------------------------------------------------------- features ---

def test_every_feature_is_produced():
    assert set(extract_features(record("ঢাকা", "রাজধানী ঢাকা।"))) == set(FEATURE_NAMES)


def test_an_answer_copied_from_the_passage_is_marked_as_found():
    features = extract_features(record("ঢাকা", "বাংলাদেশের রাজধানী ঢাকা।"))
    assert features["exact_in_passage"] == 1.0
    assert features["edit_passage"] == 0.0
    assert features["token_overlap"] == 1.0


def test_an_answer_not_in_the_passage_is_marked_as_missing():
    features = extract_features(record("লন্ডন", "বাংলাদেশের রাজধানী ঢাকা।"))
    assert features["exact_in_passage"] == 0.0
    assert features["edit_passage"] > 0.0


def test_a_trailing_full_stop_does_not_hide_a_match():
    """198 real train answers end in "।" while the passage does not. Both features must agree."""
    features = extract_features(record("আরবি।", "শব্দটি আরবি ভাষা থেকে এসেছে।"))
    assert features["exact_in_passage"] == 1.0
    assert features["edit_passage"] == 0.0


def test_a_record_with_no_passage_gets_no_evidence_values():
    features = extract_features(record("245"))
    for name, expected in NO_PASSAGE.items():
        assert features[name] == expected, name
    assert features["has_context"] == 0.0


def test_no_passage_does_not_look_like_a_perfect_match():
    """0.0 would mean "found it exactly", so a missing passage must score 1.0."""
    assert extract_features(record("245"))["edit_passage"] == 1.0


def test_answer_length_and_digit_share_are_measured():
    features = extract_features(record("1971"))
    assert features["answer_len"] == 4.0
    assert features["digit_share"] == 1.0


def test_punctuation_is_not_counted_as_a_shared_word():
    """"।" appears in nearly every passage, so counting it would inflate the overlap."""
    assert extract_features(record("লন্ডন।", "রাজধানী ঢাকা।"))["token_overlap"] == 0.0


# ------------------------------------------------------------------ score ---

def test_a_perfect_prediction_scores_one():
    assert macro_f1([1, 1, 0, 0], [1, 1, 0, 0]) == 1.0


def test_always_guessing_correct_scores_badly():
    """This is why macro-F1 is used: guessing one class cannot win."""
    assert macro_f1([1, 1, 0, 0], [1, 1, 1, 1]) < 0.5
