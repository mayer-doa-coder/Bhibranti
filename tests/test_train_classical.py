"""Tests for src/train_classical.py - the first models that learn (Labs 2-3).

Run from the repository root:

    python -m pytest tests/test_train_classical.py -v

Training a real model takes minutes, so these check the pieces around it: that text is
turned into numbers correctly, that nothing from dev leaks into training, and that every
model in the list can actually be built.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from train_classical import (  # noqa: E402  (the import has to come after the path is set)
    MODELS,
    SparseFeatures,
    confidence,
    embedding_matrix,
    lab_naive_bayes,
    lab_naive_bayes_predict,
    make_classifier,
    needs_a_seed,
    out_of_vocabulary_rate,
    parts_as_text,
    uses_vectors,
)
from skipgram import SkipGram  # noqa: E402


def record(answer: str = "ঢাকা", passage: str = "বাংলাদেশের রাজধানী ঢাকা।",
           question: str = "রাজধানী কোথায়?", label: int = 1) -> dict:
    return {"candidate_answer": answer, "context": passage, "question": question,
            "label": label, "pair_id": "p1", "difficulty": "easy", "subject": "history",
            "condition": "has_context" if passage else "no_context",
            "hallucination_type": "none" if label else "entity"}


# Repeated three times on purpose: the real vectorizers ignore any word appearing in only
# one document (min_df = 2), so a handful of one-off records would leave nothing to count.
RECORDS = [record(), record("লন্ডন", label=0), record("ঢাকা শহর"),
           record("প্যারিস", label=0)] * 3


# --------------------------------------------------- text into three columns ---

def test_the_three_parts_are_kept_apart():
    """The whole point of counting each part separately: the model can then treat a word
    in the answer differently from the same word in the passage."""
    columns = parts_as_text(RECORDS, "F2", "V1")
    assert set(columns) == {"context", "question", "answer"}
    assert "রাজধানী" in columns["context"][0]
    assert columns["answer"][0].strip() == "ঢাকা"
    assert "বাংলাদেশের" not in columns["answer"][0], "the passage leaked into the answer column"


def test_f1_leaves_the_passage_out_entirely():
    columns = parts_as_text(RECORDS, "F1", "V1")
    assert set(columns) == {"question", "answer"}


def test_a_record_with_no_passage_gives_an_empty_passage_column():
    columns = parts_as_text([record(passage="")], "F2", "V1")
    assert columns["context"][0].strip() == ""


# ------------------------------------------------------------ counting words ---

def test_counting_produces_one_row_per_record():
    features = SparseFeatures("bow")
    matrix = features.fit_transform(parts_as_text(RECORDS, "F2", "V1"))
    assert matrix.shape[0] == len(RECORDS)
    assert features.size > 0


def test_dev_is_only_transformed_never_learned_from():
    """The vectorizer must be fitted on train alone. If dev words could add columns, the
    two matrices would not line up - and the model would have seen dev during training."""
    features = SparseFeatures("tfidf")
    train = features.fit_transform(parts_as_text(RECORDS, "F2", "V1"))
    dev = features.transform(parts_as_text([record("সম্পূর্ণ_নতুন_শব্দ", label=0)], "F2", "V1"))
    assert train.shape[1] == dev.shape[1], "train and dev must have the same columns"


def test_counts_are_never_negative():
    """Naive Bayes cannot accept negative numbers, so this must hold for both weightings."""
    for weighting in ("bow", "tfidf"):
        matrix = SparseFeatures(weighting).fit_transform(parts_as_text(RECORDS, "F2", "V1"))
        assert matrix.min() >= 0


def test_unseen_words_are_counted_as_out_of_vocabulary():
    train = {"answer": ["ক খ"]}
    dev = {"answer": ["ক অজানা"]}
    assert out_of_vocabulary_rate(train, dev) == 0.5


# ----------------------------------------------------------- word vectors in ---

def test_every_record_becomes_one_vector():
    vectors = SkipGram(dim=12, min_count=1, epochs=2).fit(
        [["বাংলাদেশের", "রাজধানী", "ঢাকা"], ["লন্ডন", "শহর"]])
    matrix = embedding_matrix(RECORDS, vectors, "F2", "V1", None)
    assert matrix.shape == (len(RECORDS), 12)
    assert np.isfinite(matrix).all()


# ------------------------------------------------------------- the learners ---

def test_every_model_in_the_list_can_be_built():
    for name, (family, representation, learner) in MODELS.items():
        assert family in {"M1", "M2", "M11", "M12"}, name
        if learner is not None:
            assert make_classifier(learner, 42) is not None, name


def test_an_unknown_learner_is_refused():
    try:
        make_classifier("magic", 42)
    except ValueError:
        return
    raise AssertionError("an unknown learner must raise ValueError")


def test_the_svm_is_given_enough_iterations_to_settle():
    """At the default it reported 'failed to converge' on our 216,000 features."""
    assert make_classifier("svm", 42).max_iter >= 5000


def test_naive_bayes_uses_the_labs_smoothing():
    assert make_classifier("nb", 42).alpha == 1.0


def test_models_with_randomness_are_marked_for_three_seeds():
    assert needs_a_seed("skipgram_mean_xgb") and needs_a_seed("skipgram_mean_logreg")
    assert not needs_a_seed("bow_nb"), "counting words has no randomness in it"


def test_the_models_that_need_word_vectors_are_marked():
    assert uses_vectors("skipgram_mean_logreg") and uses_vectors("similarity_logreg")
    assert not uses_vectors("bow_nb")


def test_confidence_comes_back_for_every_kind_of_learner():
    """Logistic Regression gives probabilities; an SVM only gives distance from the line.
    Both can be ranked, which is all AUC needs."""
    x = np.array([[0.0, 1.0], [1.0, 0.0], [0.2, 0.9], [0.9, 0.1]])
    y = [0, 1, 0, 1]
    for learner in ("logreg", "svm"):
        model = make_classifier(learner, 42).fit(x, y)
        scores = confidence(model, x)
        assert scores is not None and len(scores) == len(y), learner


# --------------------------------------------- our code against the library's ---

def test_the_labs_naive_bayes_learns_an_obvious_split():
    texts = ["ভাল চমৎকার সুন্দর"] * 5 + ["খারাপ বাজে ভুল"] * 5
    labels = [1] * 5 + [0] * 5
    model = lab_naive_bayes(texts, labels)
    assert lab_naive_bayes_predict(model, "ভাল সুন্দর") == 1
    assert lab_naive_bayes_predict(model, "খারাপ ভুল") == 0


def test_the_labs_naive_bayes_handles_a_word_it_never_saw():
    """Adding 1 to every count is what stops an unseen word making the whole thing zero."""
    model = lab_naive_bayes(["ভাল"] * 3 + ["খারাপ"] * 3, [1, 1, 1, 0, 0, 0])
    assert lab_naive_bayes_predict(model, "সম্পূর্ণ_অজানা") in (0, 1)
