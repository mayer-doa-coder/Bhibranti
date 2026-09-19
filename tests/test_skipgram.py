"""Tests for src/skipgram.py - the word vectors written out from Lab 3.

Run from the repository root:

    python -m pytest tests/test_skipgram.py -v

These train tiny models on made-up sentences, so they finish in seconds. Whether the real
vectors are any good is a different question, answered by `python src/skipgram.py --check`.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from skipgram import (  # noqa: E402  (the import has to come after the path is set)
    SkipGram,
    inverse_document_frequency,
    sigmoid,
)

# Two clear groups of words. "রাজা" and "রানি" always appear with royal words, "নদী" and
# "সাগর" always with water words - so a working model should put each group together.
ROYAL = ["রাজা", "রানি", "রাজ্য", "সিংহাসন", "মুকুট"]
WATER = ["নদী", "সাগর", "পানি", "ঢেউ", "নৌকা"]
DOCUMENTS = ([ROYAL[:] for _ in range(60)] + [WATER[:] for _ in range(60)])


def tiny_model(seed: int = 42, epochs: int = 20) -> SkipGram:
    return SkipGram(dim=24, window=3, min_count=1, negatives=3, epochs=epochs,
                    seed=seed).fit(DOCUMENTS)


# ----------------------------------------------------------------- the maths ---

def test_the_squashing_function_stays_between_zero_and_one():
    values = sigmoid(np.array([-1000.0, -1.0, 0.0, 1.0, 1000.0]))
    assert np.all((values >= 0.0) & (values <= 1.0))
    assert round(float(sigmoid(np.array([0.0]))[0]), 6) == 0.5


def test_huge_numbers_do_not_break_it():
    """Without clipping, exp() of a large number overflows and the training goes to NaN."""
    assert np.all(np.isfinite(sigmoid(np.array([-1e9, 1e9]))))


# ------------------------------------------------------------- the vocabulary ---

def test_rare_words_are_dropped():
    documents = [["ক", "খ", "গ"]] * 5 + [["বিরল"]]          # "বিরল" appears once
    model = SkipGram(dim=8, min_count=2, epochs=1).fit(documents)
    assert "ক" in model and "বিরল" not in model


def test_every_word_gets_its_own_vector():
    model = tiny_model(epochs=2)
    assert model.vectors.shape == (len(model.vocab), model.dim)
    assert not np.allclose(model["রাজা"], model["নদী"]), "different words, different vectors"


def test_training_is_repeatable():
    assert np.allclose(tiny_model(42, 3).vectors, tiny_model(42, 3).vectors)


def test_a_different_seed_gives_different_vectors():
    """This is why each seed needs its own vectors: the training itself is random."""
    assert not np.allclose(tiny_model(42, 3).vectors, tiny_model(1337, 3).vectors)


def test_empty_text_is_refused_rather_than_silently_producing_nothing():
    try:
        SkipGram(min_count=5).fit([["ক"]])
    except ValueError:
        return
    raise AssertionError("training on text with no usable words must raise")


# ------------------------------------------------------------ does it learn? ---

def test_words_used_together_end_up_close():
    """The whole point: royal words near royal words, water words near water words."""
    model = tiny_model()
    royal_pair = float(np.dot(model["রাজা"], model["সিংহাসন"]))
    across = float(np.dot(model["রাজা"], model["নদী"]))
    assert royal_pair > across, "a royal word should sit closer to another royal word"


def test_the_neighbour_list_finds_the_right_group():
    """Needs more passes than the other tests: 120 sentences is a very small world, and
    the two groups only pull apart once the model has seen them many times over."""
    model = tiny_model(epochs=150)
    neighbours = [w for w, _ in model.nearest("নদী", 4)]
    assert sum(w in WATER for w in neighbours) >= 3, f"expected water words, got {neighbours}"


def test_more_training_separates_the_groups_further():
    """The real proof that it is learning, rather than one lucky neighbour list."""
    def gap(model):
        def cosine(a, b):
            va, vb = model[a], model[b]
            return float(np.dot(va, vb) / (np.linalg.norm(va) * np.linalg.norm(vb) + 1e-9))
        same = np.mean([cosine(a, b) for group in (ROYAL, WATER)
                        for a in group for b in group if a != b])
        different = np.mean([cosine(a, b) for a in ROYAL for b in WATER])
        return same - different

    assert gap(tiny_model(epochs=60)) > gap(tiny_model(epochs=5))


def test_a_word_it_never_saw_has_no_neighbours():
    assert tiny_model(epochs=2).nearest("অজানা") == []


# ------------------------------------------------------- one vector per text ---

def test_a_documents_vector_is_the_average_of_its_words():
    model = tiny_model(epochs=2)
    expected = (model["রাজা"] + model["রানি"]) / 2
    assert np.allclose(model.document_vector(["রাজা", "রানি"]), expected)


def test_unknown_words_are_skipped_not_counted_as_zero():
    model = tiny_model(epochs=2)
    assert np.allclose(model.document_vector(["রাজা", "কখনওদেখিনি"]), model["রাজা"])


def test_text_with_no_known_words_gives_a_zero_vector():
    model = tiny_model(epochs=2)
    assert np.allclose(model.document_vector(["কখনওদেখিনি"]), np.zeros(model.dim))


def test_weighting_changes_the_result():
    """A rare, informative word should pull the vector more than a common one."""
    model = tiny_model(epochs=2)
    idf = {"রাজা": 0.1, "রানি": 5.0}
    plain = model.document_vector(["রাজা", "রানি"])
    weighted = model.document_vector(["রাজা", "রানি"], idf)
    assert not np.allclose(plain, weighted)


def test_rarer_words_get_a_bigger_idf():
    documents = [["সবখানে", "বিরল"]] + [["সবখানে"]] * 9
    idf = inverse_document_frequency(documents)
    assert idf["বিরল"] > idf["সবখানে"]


def test_coverage_counts_the_words_it_knows():
    model = tiny_model(epochs=2)
    assert model.coverage([["রাজা", "কখনওদেখিনি"]]) == 0.5
    assert model.coverage([[]]) == 0.0


# --------------------------------------------------------------- saving them ---

def test_saved_vectors_come_back_the_same(tmp_path):
    model = tiny_model(epochs=2)
    reloaded = SkipGram.load(model.save(tmp_path / "v.npz"))
    assert np.allclose(model["রাজা"], reloaded["রাজা"])
    assert model.vocab == reloaded.vocab
    assert reloaded.dim == model.dim
