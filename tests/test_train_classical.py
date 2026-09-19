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
    ABLATION_MODELS,
    ABLATION_VARIANTS,
    MODELS,
    VARIANT_MEANING,
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


# ------------------------------------------- M10: does the cleanup help? ---

def test_the_ablation_covers_every_variant_the_guide_lists():
    assert ABLATION_VARIANTS == ["V0", "V1", "V2", "V3", "V4", "V2-demo"]


def test_every_variant_has_a_plain_english_meaning():
    """The table is meant to be readable by someone who has not seen the code."""
    for variant in ABLATION_VARIANTS:
        assert VARIANT_MEANING[variant], variant


def test_the_ablation_uses_real_models_from_the_list():
    for name in ABLATION_MODELS:
        assert name in MODELS, name


def test_the_ablation_covers_both_families_that_depend_on_preprocessing():
    families = {MODELS[name][0] for name in ABLATION_MODELS}
    assert families == {"M1", "M2"}, (
        "M1 counts words and M2 learns word vectors, so both change when the words change. "
        "M11 and M12 read the cleaned text directly and are unaffected.")


def test_each_variant_changes_something_specific():
    """If two variants gave identical words, comparing them would prove nothing.

    The sentence is built to exercise all three steps: "এবং" is a common word, "কলেজের"
    can be stemmed to "কলেজ", and "নয়" is a negation that only V2-demo throws away.
    """
    from preprocess import as_tokens
    sentence = record(answer="কলেজের ছাত্ররা এবং বইটি বিভক্ত নয়", passage="",
                      question="কারা পড়েনি?")
    words = {v: as_tokens(sentence, "F1", v) for v in ABLATION_VARIANTS}

    assert words["V0"] != words["V1"], "cleaning and tokenizing must change something"
    assert words["V2"] != words["V1"], "dropping common words must remove এবং"
    assert words["V3"] != words["V1"], "stemming must shorten কলেজের"
    assert words["V4"] != words["V2"], "V4 stems on top of V2"
    assert words["V2-demo"] != words["V2"], "only the demo variant throws away নয়"


def test_the_demo_variant_is_the_one_that_loses_negation():
    """V2-demo exists to show the damage; V2 must keep negation, V2-demo must not."""
    from preprocess import as_tokens
    from text_bn import clean
    sentence = record(answer="বিভক্ত নয়", passage="", question="কী?")
    assert clean("নয়") in as_tokens(sentence, "F1", "V2")
    assert clean("নয়") not in as_tokens(sentence, "F1", "V2-demo")


def test_no_separator_markers_leak_into_the_columns():
    """Each column holds ONE part, so the between-parts marker has no business there.

    It used to: every passage column ended "... <SEP> <SEP>" and every question column
    "... <SEP>", because each part was routed through the flat-sequence builder. Those
    markers then became features of their own, in every single document.
    """
    from preprocess import SEPARATOR
    for fmt in ("F1", "F2"):
        for column, texts in parts_as_text(RECORDS, fmt, "V1").items():
            assert all(SEPARATOR not in t for t in texts), f"{fmt}/{column} still has markers"


def test_a_column_holds_only_its_own_part():
    columns = parts_as_text([record(answer="ঢাকা", question="কোথায়?",
                                    passage="বাংলাদেশের রাজধানী")], "F2", "V1")
    assert columns["answer"][0].split() == ["ঢাকা"]
    assert "কোথায়" not in columns["context"][0]
    assert "রাজধানী" not in columns["question"][0]


# ------------------------------------------- a warning on one seed is not lost ---

def test_a_convergence_warning_on_any_seed_survives_the_summary(monkeypatch):
    """The failure this catches is silent by nature.

    `run_one` hands back the notes for one seed at a time. If the summary simply kept the
    LAST seed's notes, a model that failed to converge on seed 42 but converged on seed
    2024 would be printed as clean - and "did not converge" is the difference between a
    real score and a meaningless one.
    """
    import types

    import evaluate as ev
    import train_classical as tc

    calls = {"n": 0}

    def fake_run_one(name, seed, fmt, variant, vectors, quiet=False):
        calls["n"] += 1
        notes = {"trained in": "0.1s"}
        if seed == 42:                                   # only the FIRST seed complains
            notes["WARNING"] = "svm did not converge"
        return types.SimpleNamespace(macro_f1=0.5), 0.5, notes

    monkeypatch.setattr(tc, "run_one", fake_run_one)
    monkeypatch.setattr(tc, "load_or_train_vectors", lambda seed, variant: None)
    monkeypatch.setattr(tc, "MODELS", {"toy": ("M2", "skipgram_mean", "xgb")})
    monkeypatch.setattr(tc, "load_split", lambda split: [])
    monkeypatch.setattr(ev, "baselines_for", lambda records: {})

    captured = []
    monkeypatch.setattr("builtins.print", lambda *a, **k: captured.append(" ".join(map(str, a))))
    tc.run_all((42, 1337, 2024), "F2", "V1", write_log=False)

    assert calls["n"] == 3, "a model with randomness must run on all three seeds"
    assert any("did not converge" in line for line in captured), (
        "the seed-42 warning was swallowed by the later seeds")
    assert any("42" in line for line in captured if "did not converge" in line), (
        "the report should say WHICH seed failed to converge")
