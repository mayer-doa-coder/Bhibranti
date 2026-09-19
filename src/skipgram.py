"""Word vectors, learned from our own text: NLP Lab 3's Skip-gram, written out (M2).

A bag of words treats "ঢাকা" and "রাজধানী" as two unrelated symbols. Skip-gram fixes that
by giving every word a list of numbers, learned from the company it keeps, so words used
in similar places end up with similar numbers.

    from skipgram import SkipGram
    vectors = SkipGram().fit(documents)        # documents = lists of tokens
    vectors.nearest("নদী")                     # the words it thinks are related
    vectors.document_vector(tokens)            # one vector for a whole answer

Run from the repository root:

    python src/skipgram.py --train     # train on the train split and save the vectors
    python src/skipgram.py --check     # the Lab 3 sanity check: are the neighbours sensible?

WHY THIS IS WRITTEN OUT INSTEAD OF INSTALLED
The guide names gensim, but gensim has no build for the Python version this project runs
on (3.14). Lab 3 writes Skip-gram from scratch anyway, so it is written out here - which
also means every step is visible rather than hidden inside a library.

ONE CHANGE FROM THE LAB'S VERSION, AND WHY
Lab 3 scores every word in the vocabulary for each training pair (a softmax). That costs
one pass over 17,701 words per pair, and this corpus produces about 1.7 million pairs per
round - hours of work for one training run.

Instead this uses **negative sampling** (the standard trick from the original word2vec
paper): for each real (word, neighbour) pair, pick a handful of random words that were NOT
the neighbour, and train the model to tell the real one from the fakes. Same idea - words
that appear together end up close - but a few comparisons per pair instead of 17,701.

THE OTHER PRACTICAL TOUCH: IGNORING VERY COMMON WORDS SOMETIMES
The top 20 tokens are 24% of all our text ("।", "এবং", "-", ...). They teach almost
nothing, so each one is randomly skipped in proportion to how common it is. This is
word2vec's "subsampling", and it both speeds training up and improves the vectors.
"""

from __future__ import annotations

import argparse
import collections
import json
import math
import sys
from pathlib import Path

import numpy as np

# =============================================================================
# SETTINGS - the guide's values (§7.1), plus the two that negative sampling needs
# =============================================================================

VECTOR_SIZE = 200      # numbers per word
WINDOW = 5             # how far to look either side of a word
MIN_COUNT = 2          # ignore words seen fewer times than this (17,701 words survive)
NEGATIVES = 5          # fake neighbours per real one
EPOCHS = 5             # passes over the text
START_LEARNING_RATE = 0.025
FINAL_LEARNING_RATE = 0.0001
SUBSAMPLE = 1e-3       # how aggressively very common words are skipped
BATCH = 8192           # pairs updated at once, so numpy does the work instead of Python

VECTORS_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"


def vectors_path(seed: int = 42, variant: str = "V1") -> Path:
    """Where one set of vectors lives.

    The seed and the variant are both in the name because both change the vectors:
    training starts from random numbers, and stemming changes the words themselves.
    Every reader and writer in the project uses this one function, so they cannot
    disagree about the filename.
    """
    return VECTORS_DIR / f"skipgram_s{seed}_{variant}.npz"


VECTORS_FILE = vectors_path()          # the default set: seed 42, variant V1


def sigmoid(x: np.ndarray) -> np.ndarray:
    """Squash any number into 0..1. Clipped so exp() cannot overflow."""
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


class SkipGram:
    """Learns a vector per word by predicting the words around it."""

    def __init__(self, dim: int = VECTOR_SIZE, window: int = WINDOW,
                 min_count: int = MIN_COUNT, negatives: int = NEGATIVES,
                 epochs: int = EPOCHS, seed: int = 42) -> None:
        self.dim = dim
        self.window = window
        self.min_count = min_count
        self.negatives = negatives
        self.epochs = epochs
        self.seed = seed
        self.vocab: dict[str, int] = {}
        self.counts: np.ndarray = np.zeros(0)
        self.vectors: np.ndarray = np.zeros((0, dim))

    # ---------------------------------------------------------------- setup ---

    def _build_vocabulary(self, documents: list[list[str]]) -> np.ndarray:
        counts = collections.Counter(token for document in documents for token in document)
        kept = sorted((w for w, c in counts.items() if c >= self.min_count),
                      key=lambda w: (-counts[w], w))          # sorted, so runs are repeatable
        self.vocab = {word: i for i, word in enumerate(kept)}
        self.counts = np.array([counts[w] for w in kept], dtype=np.float64)
        return self.counts

    def _keep_probability(self) -> np.ndarray:
        """Chance of keeping each word - very common words are usually skipped."""
        frequency = self.counts / self.counts.sum()
        with np.errstate(divide="ignore"):
            keep = (np.sqrt(frequency / SUBSAMPLE) + 1) * (SUBSAMPLE / frequency)
        return np.clip(keep, 0.0, 1.0)

    def _sampling_table(self) -> np.ndarray:
        """Probabilities for drawing fake neighbours.

        Counts are raised to the power 0.75, the word2vec recipe: it pulls rare words up a
        little so they get used as negatives too, without letting them dominate.
        """
        weights = self.counts ** 0.75
        return weights / weights.sum()

    # ------------------------------------------------------------- training ---

    def fit(self, documents: list[list[str]], verbose: bool = False) -> SkipGram:
        rng = np.random.default_rng(self.seed)
        self._build_vocabulary(documents)
        if not self.vocab:
            raise ValueError("no words survived min_count - is the text empty?")

        keep_probability = self._keep_probability()
        negative_probability = self._sampling_table()
        as_ids = [np.array([self.vocab[t] for t in doc if t in self.vocab], dtype=np.int32)
                  for doc in documents]

        # Two sets of numbers per word: one for when it is the centre word (these are the
        # vectors we keep), one for when it is a neighbour (thrown away after training).
        self.vectors = (rng.random((len(self.vocab), self.dim), dtype=np.float32) - 0.5) / self.dim
        neighbour_vectors = np.zeros((len(self.vocab), self.dim), dtype=np.float32)

        total_batches = 0
        for epoch in range(self.epochs):
            centres, contexts = self._pairs_for_one_pass(as_ids, keep_probability, rng)
            if len(centres) == 0:
                continue
            order = rng.permutation(len(centres))
            centres, contexts = centres[order], contexts[order]

            for start in range(0, len(centres), BATCH):
                progress = (epoch + start / max(len(centres), 1)) / self.epochs
                rate = START_LEARNING_RATE + progress * (FINAL_LEARNING_RATE - START_LEARNING_RATE)
                self._update(centres[start:start + BATCH], contexts[start:start + BATCH],
                             neighbour_vectors, negative_probability, rate, rng)
                total_batches += 1
            if verbose:
                print(f"    pass {epoch + 1}/{self.epochs}: {len(centres):,} pairs")
        if verbose:
            print(f"    {total_batches:,} batches over {len(self.vocab):,} words")
        return self

    def _pairs_for_one_pass(self, documents: list[np.ndarray], keep_probability: np.ndarray,
                            rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        """Every (word, nearby word) pair for one pass, after skipping common words.

        The window is re-drawn for each centre word, between 1 and `window`. That is
        word2vec's way of weighting near neighbours more than distant ones.
        """
        centres, contexts = [], []
        for document in documents:
            if len(document) < 2:
                continue
            kept = document[rng.random(len(document)) < keep_probability[document]]
            if len(kept) < 2:
                continue
            widths = rng.integers(1, self.window + 1, size=len(kept))
            for position, (word, width) in enumerate(zip(kept, widths)):
                low = max(0, position - width)
                high = min(len(kept), position + width + 1)
                neighbours = np.concatenate([kept[low:position], kept[position + 1:high]])
                if len(neighbours):
                    centres.append(np.full(len(neighbours), word, dtype=np.int32))
                    contexts.append(neighbours)
        if not centres:
            return np.zeros(0, np.int32), np.zeros(0, np.int32)
        return np.concatenate(centres), np.concatenate(contexts)

    def _update(self, centres: np.ndarray, contexts: np.ndarray, neighbour_vectors: np.ndarray,
                negative_probability: np.ndarray, rate: float, rng: np.random.Generator) -> None:
        """One batch of learning: pull real pairs together, push fake ones apart."""
        fakes = rng.choice(len(self.vocab), size=(len(centres), self.negatives),
                           p=negative_probability)

        centre = self.vectors[centres]                       # (B, D)
        real = neighbour_vectors[contexts]                   # (B, D)
        fake = neighbour_vectors[fakes]                      # (B, K, D)

        # How strongly the model already believes each pairing. We want ~1 for the real
        # neighbour and ~0 for the fakes; the gap is the error to correct.
        real_error = sigmoid(np.sum(centre * real, axis=1)) - 1.0            # (B,)
        fake_error = sigmoid(np.einsum("bkd,bd->bk", fake, centre))          # (B, K)

        centre_gradient = (real_error[:, None] * real
                           + np.einsum("bk,bkd->bd", fake_error, fake))
        real_gradient = real_error[:, None] * centre
        fake_gradient = fake_error[..., None] * centre[:, None, :]

        # np.add.at, not -=, because the same word can appear several times in one batch
        # and every one of those updates has to count.
        np.add.at(self.vectors, centres, -rate * centre_gradient)
        np.add.at(neighbour_vectors, contexts, -rate * real_gradient)
        np.add.at(neighbour_vectors, fakes.ravel(),
                  -rate * fake_gradient.reshape(-1, self.dim))

    # ---------------------------------------------------------------- using ---

    def __contains__(self, word: str) -> bool:
        return word in self.vocab

    def __getitem__(self, word: str) -> np.ndarray:
        return self.vectors[self.vocab[word]]

    def coverage(self, documents: list[list[str]]) -> float:
        """Share of tokens the vectors actually know - logged for M2 (guide §7.1)."""
        total = sum(len(d) for d in documents)
        known = sum(1 for d in documents for t in d if t in self.vocab)
        return known / total if total else 0.0

    def document_vector(self, tokens: list[str], idf: dict[str, float] | None = None) -> np.ndarray:
        """One vector for a whole piece of text (Lab 3, Module 4).

        Without `idf`: the plain average of the word vectors.
        With `idf`: a weighted average, so a rare, informative word counts for more than
        "এবং". Both are required by M2, and the two are compared in the results.
        """
        known = [t for t in tokens if t in self.vocab]
        if not known:
            return np.zeros(self.dim, dtype=np.float32)
        rows = self.vectors[[self.vocab[t] for t in known]]
        if idf is None:
            return rows.mean(axis=0)
        counts = collections.Counter(known)
        weights = np.array([(counts[t] / len(known)) * idf.get(t, 1.0) for t in known],
                           dtype=np.float32)
        if weights.sum() <= 0:
            return rows.mean(axis=0)
        return (rows * weights[:, None]).sum(axis=0) / weights.sum()

    def nearest(self, word: str, n: int = 5) -> list[tuple[str, float]]:
        """The n closest words by cosine similarity - the Lab 3 sanity check."""
        if word not in self.vocab:
            return []
        lengths = np.linalg.norm(self.vectors, axis=1) + 1e-9
        similarity = (self.vectors @ self[word]) / (lengths * (np.linalg.norm(self[word]) + 1e-9))
        best = np.argsort(-similarity)[:n + 1]
        words = list(self.vocab)
        return [(words[i], float(similarity[i])) for i in best if words[i] != word][:n]

    # --------------------------------------------------------------- saving ---

    def save(self, path: Path = VECTORS_FILE) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, vectors=self.vectors,
                            words=np.array(list(self.vocab), dtype=object),
                            settings=json.dumps({"dim": self.dim, "window": self.window,
                                                 "min_count": self.min_count,
                                                 "negatives": self.negatives,
                                                 "epochs": self.epochs, "seed": self.seed}))
        return path

    @classmethod
    def load(cls, path: Path = VECTORS_FILE) -> SkipGram:
        data = np.load(path, allow_pickle=True)
        settings = json.loads(str(data["settings"]))
        model = cls(**settings)
        model.vectors = data["vectors"]
        model.vocab = {w: i for i, w in enumerate(data["words"].tolist())}
        return model


def inverse_document_frequency(documents: list[list[str]]) -> dict[str, float]:
    """How rare each word is, for the weighted document vector (Lab 2's IDF)."""
    seen: collections.Counter = collections.Counter()
    for document in documents:
        seen.update(set(document))
    total = len(documents)
    return {word: math.log((1 + total) / (1 + n)) + 1 for word, n in seen.items()}


# =============================================================================
# COMMAND LINE
# =============================================================================

def train_on_split(seed: int = 42, verbose: bool = True, variant: str = "V1") -> SkipGram:
    """Train on the TRAIN split only - dev and test never shape the vectors.

    `variant` must match the preprocessing the vectors will later be used with. Vectors
    learned on whole words know nothing about "কলেজ" once stemming has turned "কলেজের"
    into it, so the M10 experiment trains a fresh set for every variant.
    """
    from preprocess import as_tokens
    from splits import load_split

    documents = [as_tokens(r, "F2", variant) for r in load_split("train")]
    if verbose:
        print(f"  {len(documents):,} documents, "
              f"{sum(len(d) for d in documents):,} tokens, {variant}, seed {seed}")
    return SkipGram(seed=seed).fit(documents, verbose=verbose)


def run_check() -> int:
    """Lab 3's sanity check: if the neighbours look random, the vectors are not usable."""
    from preprocess import as_tokens
    from splits import load_split

    path = vectors_path()
    model = SkipGram.load(path) if path.exists() else train_on_split()
    dev_documents = [as_tokens(r, "F2") for r in load_split("dev")]

    print("=" * 78)
    print(f"SKIP-GRAM CHECK - {len(model.vocab):,} words, {model.dim} numbers each")
    print("=" * 78)
    print(f"  words of dev text the vectors know: {model.coverage(dev_documents):.1%}")
    print("\n  Nearest words (the test is whether these look related to a Bangla speaker):")
    for word in ["নদী", "রাজা", "সাল", "বিজ্ঞান", "ঢাকা", "যুদ্ধ", "ভাষা", "কবি", "1971", "সরকার"]:
        neighbours = model.nearest(word, 5)
        if neighbours:
            print(f"    {word:<12} {', '.join(f'{w} ({s:.2f})' for w, s in neighbours)}")
        else:
            print(f"    {word:<12} (not in the vocabulary)")

    print("\n  A small corpus like ours (345k words) gives weaker vectors than the millions")
    print("  word2vec is normally trained on. Judge them by whether they help the model,")
    print("  not by whether every neighbour is perfect.")
    print("=" * 78)
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Skip-gram word vectors (Lab 3, M2).")
    parser.add_argument("--train", action="store_true", help="train and save the vectors")
    parser.add_argument("--check", action="store_true", help="are the neighbours sensible?")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--variant", default="V1",
                        help="which text preparation the vectors should learn from")
    args = parser.parse_args()

    if args.train:
        model = train_on_split(args.seed, variant=args.variant)
        print(f"  saved to {model.save(vectors_path(args.seed, args.variant))}")
        return 0
    if args.check:
        return run_check()
    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
