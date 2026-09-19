"""Tests for src/evaluate.py - one test per rule, each with a plain-language name.

Run from the repository root:

    python -m pytest tests/test_evaluate.py -v

Scoring code is the last place a silent mistake should live: if it is wrong, every
model's number is wrong and nothing downstream can catch it. So the basics are checked
against values worked out by hand.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from evaluate import (  # noqa: E402  (the import has to come after the path is set)
    accuracy,
    auc,
    baseline_predictions,
    bootstrap_interval,
    by_error_type,
    confusion,
    evaluate,
    macro_f1,
    mcnemar,
    precision_recall_f1,
    shuffle_tokens,
    shuffled_record,
    slice_by,
    summarise_seeds,
    tune_threshold,
    write_table,
)


def record(label: int, pair: str, condition: str = "has_context", difficulty: str = "easy",
           subject: str = "history", kind: str = "none") -> dict:
    return {
        "label": label,
        "pair_id": pair,
        "condition": condition,
        "difficulty": difficulty,
        "subject": subject,
        "hallucination_type": kind,
        "context": "বাংলাদেশের রাজধানী ঢাকা।" if condition == "has_context" else "",
        "question": "রাজধানী কোথায়?",
        "candidate_answer": "ঢাকা" if label == 1 else "লন্ডন",
    }


# --------------------------------------------------------------- the basics ---

def test_a_perfect_prediction_scores_one():
    assert macro_f1([1, 1, 0, 0], [1, 1, 0, 0]) == 1.0


def test_always_answering_correct_scores_one_third():
    """Worked out by hand: the 'correct' class gets F1 0.667, the other 0, so 0.333.

    This is the whole reason macro-F1 is the main number - accuracy would say 0.50.
    """
    assert round(macro_f1([1, 1, 0, 0], [1, 1, 1, 1]), 3) == 0.333
    assert accuracy([1, 1, 0, 0], [1, 1, 1, 1]) == 0.5


def test_getting_everything_backwards_scores_zero():
    assert macro_f1([1, 1, 0, 0], [0, 0, 1, 1]) == 0.0


def test_precision_and_recall_match_a_hand_worked_example():
    truth = [0, 0, 0, 1]                  # three hallucinations, one good answer
    predicted = [0, 0, 1, 1]              # two of the three caught, the third let through
    precision, recall, f1 = precision_recall_f1(truth, predicted, 0)
    assert precision == 1.0               # everything it flagged really was wrong...
    assert round(recall, 3) == 0.667      # ...but it only found two of the three
    assert round(f1, 3) == 0.8


def test_the_confusion_counts_add_up():
    """Counting hallucinations (label 0): one caught, one false alarm, one missed.

    truth     0  0  1  1
    predicted 0  1  0  1
              ^  ^  ^
              |  |  called it a hallucination when the answer was fine  -> false alarm
              |  let a hallucination through                            -> missed
              caught a hallucination                                    -> hit
    """
    hit, false_alarm, missed = confusion([0, 0, 1, 1], [0, 1, 0, 1], 0)
    assert (hit, false_alarm, missed) == (1, 1, 1)


def test_auc_is_one_when_the_ranking_is_perfect():
    assert auc([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == 1.0


def test_auc_is_zero_when_the_ranking_is_upside_down():
    assert auc([1, 1, 0, 0], [0.1, 0.2, 0.8, 0.9]) == 0.0


def test_auc_is_a_half_when_every_score_is_the_same():
    assert auc([1, 0, 1, 0], [0.5, 0.5, 0.5, 0.5]) == 0.5


def test_auc_needs_confidence_scores():
    assert auc([1, 0], None) is None


# ---------------------------------------------------------------- one slice ---

def test_a_result_reports_what_was_caught_and_missed():
    records = [record(1, "p1"), record(0, "p1"), record(0, "p2"), record(1, "p2")]
    result = evaluate(records, [1, 0, 1, 1])        # one hallucination missed
    assert result.n == 4
    assert result.counts["hallucinations caught"] == 1
    assert result.counts["hallucinations missed"] == 1
    assert result.counts["false alarms"] == 0


def test_small_slices_are_flagged():
    records = [record(1, "p1"), record(0, "p1")]
    assert evaluate(records, [1, 0]).is_small, "2 records is far too few to read a score from"


# ------------------------------------------------------------- the breakdown ---

def test_slices_are_grouped_and_the_biggest_comes_first():
    records = ([record(1, "p1", subject="history"), record(0, "p1", subject="history")]
               + [record(1, "p2", subject="history"), record(0, "p2", subject="history")]
               + [record(1, "p3", subject="law"), record(0, "p3", subject="law")])
    results = slice_by(records, [1, 0] * 3, None, "subject")
    assert [r.name for r in results] == ["history", "law"]
    assert [r.n for r in results] == [4, 2]


def test_error_types_report_what_share_was_caught():
    records = [record(1, "p1"), record(0, "p1", kind="numeric"),
               record(1, "p2"), record(0, "p2", kind="numeric")]
    rows = by_error_type(records, [1, 0, 1, 1])      # one of the two numeric errors caught
    assert rows[0][0] == "numeric"
    assert rows[0][1] == 2 and rows[0][2] == 0.5


def test_untyped_answers_are_left_out_of_the_type_table():
    records = [record(1, "p1"), record(0, "p1", kind="unlabeled")]
    assert by_error_type(records, [1, 0]) == []


# --------------------------------------------------------------- statistics ---

def test_the_confidence_interval_brackets_the_score():
    records = [record(i % 2, f"p{i // 2}") for i in range(40)]
    predicted = [r["label"] for r in records]
    low, high = bootstrap_interval(records, predicted, samples=200)
    assert low <= 1.0 <= high


def test_the_confidence_interval_is_the_same_every_run():
    records = [record(i % 2, f"p{i // 2}") for i in range(40)]
    predicted = [1] * 40
    assert bootstrap_interval(records, predicted, samples=100) == \
           bootstrap_interval(records, predicted, samples=100)


def test_the_confidence_interval_draws_whole_pairs():
    """Both answers to a question must be drawn together, never split apart.

    With every pair identical, any honest pair-level resample gives the same score, so
    the interval collapses to a point. Resampling single records would not do that.
    """
    records = [record(i % 2, f"p{i // 2}") for i in range(40)]
    predicted = [r["label"] for r in records]
    low, high = bootstrap_interval(records, predicted, samples=100)
    assert low == high == 1.0


def test_two_identical_models_are_not_called_different():
    predicted = [1, 0, 1, 0]
    a_only, b_only, p = mcnemar([1, 0, 1, 0], predicted, predicted)
    assert (a_only, b_only, p) == (0, 0, 1.0)


def test_a_one_sided_win_gets_a_small_p_value():
    truth = [1] * 10
    better = [1] * 10
    worse = [0] * 10
    a_only, b_only, p = mcnemar(truth, better, worse)
    assert a_only == 10 and b_only == 0
    assert p < 0.01, "ten wins to nil should not look like chance"


def test_seed_results_are_summarised_as_mean_and_spread():
    mean, spread = summarise_seeds([0.80, 0.82, 0.84])
    assert round(mean, 3) == 0.82 and spread > 0


def test_threshold_tuning_finds_the_obvious_split():
    truth = [0, 0, 1, 1]
    scores = [0.1, 0.2, 0.8, 0.9]
    threshold, score = tune_threshold(truth, scores)
    assert score == 1.0 and 0.2 < threshold <= 0.8


# ------------------------------------------------------- the word-order test ---

def test_shuffling_keeps_every_word():
    tokens = ["ঢাকা", "বাংলাদেশের", "রাজধানী", "।"]
    assert sorted(shuffle_tokens(tokens)) == sorted(tokens)


def test_shuffling_is_the_same_every_run():
    tokens = list("abcdefgh")
    assert shuffle_tokens(tokens, 42) == shuffle_tokens(tokens, 42)


def test_shuffling_does_not_move_words_between_the_parts():
    """The passage's words must stay in the passage - only the order inside it changes."""
    original = record(1, "p1")
    original["context"] = "আলাওল পঞ্চাশ দিন কারাভোগ করেন"
    original["question"] = "কতদিন কারাভোগ করেন"
    jumbled = shuffled_record(original)
    assert sorted(jumbled["context"].split()) == sorted(original["context"].split())
    assert "আলাওল" not in jumbled["question"], "a passage word leaked into the question"


def test_shuffling_leaves_the_label_alone():
    original = record(0, "p1")
    assert shuffled_record(original)["label"] == original["label"]


# ---------------------------------------------------------------- baselines ---

def test_the_exact_rule_finds_an_answer_that_is_in_the_passage():
    records = [record(1, "p1"), record(0, "p1")]       # "ঢাকা" is in the passage, "লন্ডন" is not
    assert baseline_predictions(records, "exact") == [1, 0]


def test_a_record_with_no_passage_is_called_wrong_by_the_rules():
    records = [record(1, "p1", condition="no_context")]
    assert baseline_predictions(records, "exact") == [0]


def test_a_results_table_can_be_written(tmp_path):
    import evaluate as ev
    ev.TABLES_DIR = tmp_path
    path = write_table([{"model": "x", "macro_f1": 0.5}], "t.csv")
    assert path.exists() and "macro_f1" in path.read_text(encoding="utf-8")


# ------------------------------------------------------- the experiment log ---

def test_the_log_is_created_with_its_header_if_it_is_missing(tmp_path):
    """A finished training run must not be lost to a missing folder.

    This is the last thing a run does, after the model is already trained, so a crash
    here throws away the whole run.
    """
    import csv
    import evaluate as ev
    original = ev.LOG_FILE
    try:
        ev.LOG_FILE = tmp_path / "results" / "experiment_log.csv"   # folder does not exist
        path = ev.ensure_log_file()
        assert path.exists()
        assert next(csv.reader(path.open(encoding="utf-8"))) == ev.LOG_HEADER
    finally:
        ev.LOG_FILE = original


def test_an_existing_log_is_never_overwritten(tmp_path):
    import evaluate as ev
    original = ev.LOG_FILE
    try:
        ev.LOG_FILE = tmp_path / "experiment_log.csv"
        ev.LOG_FILE.write_text("run_id,date\nkeep_me,2026-09-19\n", encoding="utf-8")
        ev.ensure_log_file()
        assert "keep_me" in ev.LOG_FILE.read_text(encoding="utf-8")
    finally:
        ev.LOG_FILE = original


def test_the_header_matches_the_real_log_on_disk():
    """If someone adds a column to the log by hand, ensure_log_file() would write a
    different header on the next fresh clone. This catches that drift."""
    import csv
    import evaluate as ev
    if ev.LOG_FILE.exists():
        header = next(csv.reader(ev.LOG_FILE.open(encoding="utf-8")))
        assert header == ev.LOG_HEADER, "results/experiment_log.csv and LOG_HEADER disagree"


def test_an_empty_results_table_is_refused_with_a_clear_message():
    """It used to raise a bare IndexError from rows[0], at the end of a long run, with
    nothing in the message pointing at the real cause."""
    import evaluate as ev
    try:
        ev.write_table([], "should_never_be_written.csv")
    except ValueError as e:
        assert "empty" in str(e).lower()
        assert not (ev.TABLES_DIR / "should_never_be_written.csv").exists()
        return
    raise AssertionError("writing an empty table must be refused")


def test_the_log_is_read_and_written_through_one_path(tmp_path):
    """log_run resolves the path once and uses it for both the duplicate-id check and the
    append. Reading one file and appending to another is how duplicate run ids appear."""
    import csv
    import evaluate as ev
    import train_classical as tc

    original = ev.LOG_FILE
    try:
        ev.LOG_FILE = tmp_path / "experiment_log.csv"
        result = ev.Result(name="x", n=1, macro_f1=0.5, accuracy=0.5,
                           correct_class=(0.5, 0.5, 0.5), wrong_class=(0.5, 0.5, 0.5))
        for _ in range(3):
            tc.log_run("bow_nb", 42, "F2", "V1", result, 0.5, {})
        ids = [r["run_id"] for r in csv.DictReader(ev.LOG_FILE.open(encoding="utf-8"))]
        assert ids == ["m1_bow_nb_F2_V1_s42_1", "m1_bow_nb_F2_V1_s42_2", "m1_bow_nb_F2_V1_s42_3"], ids
        assert len(ids) == len(set(ids)), "run ids must be unique"
    finally:
        ev.LOG_FILE = original


def test_no_module_keeps_its_own_copy_of_the_log_path():
    """A module-level `LOG_FILE = evaluate.LOG_FILE` is bound at import and goes stale."""
    import features
    import train_classical
    assert not hasattr(train_classical, "LOG_FILE")
    assert not hasattr(features, "LOG_FILE")
