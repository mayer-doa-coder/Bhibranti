"""The first models that actually learn: Labs 2 and 3 (M1, M2, M11, M12).

Every model here trains on the train split in seconds to minutes on a plain laptop - no
GPU. They are the floor the later, heavier models have to clear.

    python src/train_classical.py --model tfidf_logreg       # train one, score it on dev
    python src/train_classical.py --all                      # all of them, side by side
    python src/train_classical.py --all --log                # ...and record them
    python src/train_classical.py --model skipgram_mean_logreg --seeds   # all three seeds
    python src/train_classical.py --model tfidf_logreg --variant V2      # M10 ablation
    python src/train_classical.py --lab-check                # our code vs the library's

WHAT IS IN HERE

  M1  counting words         Bag of Words and TF-IDF, each into Naive Bayes, Logistic
                             Regression and a linear SVM                       (Labs 2, 3)
  M11 language model         one model of correct answers, one of wrong ones      (Lab 2)
  M2  word vectors           Skip-gram averaged two ways, into Logistic Regression
                             and XGBoost                                          (Lab 3)
  M12 similarity numbers     edit distance, overlap, cosine -> Logistic Regression
                                                                            (Labs 1, 2, 3)

HOW A RECORD REACHES A COUNTING MODEL
The passage, the question and the answer are counted SEPARATELY and then joined side by
side, so the model can weigh a word differently depending on where it appeared. A word in
the answer is not the same evidence as the same word in the passage.

WHAT THESE MODELS CANNOT DO
None of them can check whether the answer matches the passage - a bag of words has no way
to line the two up. That is exactly what M11 and M12 add, and why they are here too.

EVERYTHING IS FITTED ON TRAIN ONLY
The vectorizers, the word vectors, the IDF weights and the models all see the train split
and nothing else. Dev is only ever transformed and scored, never learned from.
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
import warnings
from datetime import date
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix, hstack
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

import evaluate
from features import AnswerStyleClassifier, feature_matrix, feature_names
from preprocess import as_tokens, record_parts
from skipgram import VECTORS_FILE, SkipGram, inverse_document_frequency, train_on_split
from splits import load_split

# =============================================================================
# SETTINGS
# =============================================================================

# Word pieces and character pieces. Character pieces matter in Bengali: the same root
# appears as কলেজ / কলেজের / কলেজে, and word counting treats those as three strangers.
WORD_NGRAMS = (1, 2)
CHAR_NGRAMS = (3, 5)

# Ignore anything appearing in only one training document - it cannot generalise and it
# doubles the number of features.
MIN_DOCUMENT_FREQUENCY = 2

LOG_FILE = Path(__file__).resolve().parents[1] / "results" / "experiment_log.csv"


# =============================================================================
# PART 1 - TURNING RECORDS INTO NUMBERS
# =============================================================================

def parts_as_text(records: list[dict], fmt: str, variant: str) -> dict[str, list[str]]:
    """The records' three pieces of text, ready for counting.

    F1 leaves the passage out, so its block is simply not built.
    """
    wanted = ["question", "answer"] if fmt == "F1" else ["context", "question", "answer"]
    columns: dict[str, list[str]] = {name: [] for name in wanted}
    for record in records:
        parts = record_parts(record, raw=(variant == "V0"))
        for name in wanted:
            columns[name].append(" ".join(
                as_tokens({"context": parts.context if name == "context" else "",
                           "question": parts.question if name == "question" else "",
                           "candidate_answer": parts.answer if name == "answer" else ""},
                          "F2", variant)))
    return columns


class SparseFeatures:
    """Counts of word and character pieces, one block per part of the record."""

    def __init__(self, weighting: str = "tfidf") -> None:
        self.weighting = weighting
        self.vectorizers: dict[str, list] = {}

    def _new_vectorizer(self, analyzer: str):
        maker = TfidfVectorizer if self.weighting == "tfidf" else CountVectorizer
        grams = WORD_NGRAMS if analyzer == "word" else CHAR_NGRAMS
        return maker(analyzer=analyzer if analyzer == "word" else "char_wb",
                     ngram_range=grams, min_df=MIN_DOCUMENT_FREQUENCY, lowercase=False)

    def fit_transform(self, columns: dict[str, list[str]]) -> csr_matrix:
        blocks = []
        for part, texts in columns.items():
            self.vectorizers[part] = [self._new_vectorizer("word"), self._new_vectorizer("char")]
            blocks += [v.fit_transform(texts) for v in self.vectorizers[part]]
        return hstack(blocks).tocsr()

    def transform(self, columns: dict[str, list[str]]) -> csr_matrix:
        blocks = []
        for part, texts in columns.items():
            blocks += [v.transform(texts) for v in self.vectorizers[part]]
        return hstack(blocks).tocsr()

    @property
    def size(self) -> int:
        return sum(len(v.vocabulary_) for group in self.vectorizers.values() for v in group)


def out_of_vocabulary_rate(train_columns: dict[str, list[str]],
                           dev_columns: dict[str, list[str]]) -> float:
    """Share of dev words the training text never contained (guide asks M1 to log this)."""
    known = {w for texts in train_columns.values() for t in texts for w in t.split()}
    dev_words = [w for texts in dev_columns.values() for t in texts for w in t.split()]
    return sum(1 for w in dev_words if w not in known) / len(dev_words) if dev_words else 0.0


def embedding_matrix(records: list[dict], vectors: SkipGram, fmt: str, variant: str,
                     idf: dict[str, float] | None) -> np.ndarray:
    """One row per record: the average of its word vectors, plain or IDF-weighted."""
    return np.array([vectors.document_vector(as_tokens(r, fmt, variant), idf) for r in records])


# =============================================================================
# PART 2 - THE MODELS
# =============================================================================

def make_classifier(name: str, seed: int):
    """The learner itself. Everything with randomness is given the run's seed."""
    if name == "nb":
        return MultinomialNB(alpha=1.0)                      # Laplace smoothing, as in Lab 3
    if name == "logreg":
        return LogisticRegression(max_iter=2000, random_state=seed)
    if name == "svm":
        # 5000, not the default 1000. That is enough for TF-IDF, whose values are scaled
        # to at most 1. It is NOT enough for raw counts, which reach 55 here and do not
        # settle even at 20,000 - see the note in train_and_predict(), which reports it
        # rather than hiding it.
        return LinearSVC(random_state=seed, max_iter=5000)
    if name == "xgb":
        from xgboost import XGBClassifier
        return XGBClassifier(n_estimators=300, max_depth=5, learning_rate=0.1,
                             subsample=0.8, colsample_bytree=0.8, random_state=seed,
                             n_jobs=4, eval_metric="logloss")
    raise ValueError(f"unknown classifier {name!r}")


def confidence(model, features) -> list[float] | None:
    """The model's belief that an answer is CORRECT, for AUC and threshold tuning.

    Not every learner gives probabilities - a linear SVM only reports which side of the
    line a record fell on, and how far. Either works for ranking.
    """
    if hasattr(model, "predict_proba"):
        return list(model.predict_proba(features)[:, 1])
    if hasattr(model, "decision_function"):
        return list(model.decision_function(features))
    return None


# Every model this file can train: (family, how the text becomes numbers, the learner).
MODELS = {
    "bow_nb":                 ("M1", "bow", "nb"),
    "tfidf_nb":               ("M1", "tfidf", "nb"),
    "bow_logreg":             ("M1", "bow", "logreg"),
    "tfidf_logreg":           ("M1", "tfidf", "logreg"),
    "bow_svm":                ("M1", "bow", "svm"),
    "tfidf_svm":              ("M1", "tfidf", "svm"),
    "char_lm":                ("M11", "language_model", None),
    "skipgram_mean_logreg":   ("M2", "skipgram_mean", "logreg"),
    "skipgram_tfidf_logreg":  ("M2", "skipgram_tfidf", "logreg"),
    "skipgram_mean_xgb":      ("M2", "skipgram_mean", "xgb"),
    "skipgram_tfidf_xgb":     ("M2", "skipgram_tfidf", "xgb"),
    "similarity_logreg":      ("M12", "similarity", "logreg"),
}


def train_and_predict(name: str, train: list[dict], dev: list[dict], seed: int,
                      fmt: str, variant: str, vectors: SkipGram | None
                      ) -> tuple[list[int], list[float] | None, dict[str, str]]:
    """Fit one model on train and predict dev. Returns (predictions, confidence, notes)."""
    family, representation, learner = MODELS[name]
    notes: dict[str, str] = {}

    if representation == "language_model":
        # M11(b): one character language model per class - it reads only the answer.
        model = AnswerStyleClassifier().fit(train)
        predicted = [model.predict(r) for r in dev]
        return predicted, [model.score_gap(r) for r in dev], notes

    if representation in ("bow", "tfidf"):
        train_columns = parts_as_text(train, fmt, variant)
        dev_columns = parts_as_text(dev, fmt, variant)
        features = SparseFeatures(representation)
        x_train = features.fit_transform(train_columns)
        x_dev = features.transform(dev_columns)
        notes["features"] = f"{features.size:,}"
        notes["dev OOV"] = f"{out_of_vocabulary_rate(train_columns, dev_columns):.1%}"

    elif representation.startswith("skipgram"):
        assert vectors is not None
        idf = None
        if representation.endswith("tfidf"):
            idf = inverse_document_frequency([as_tokens(r, fmt, variant) for r in train])
        x_train = embedding_matrix(train, vectors, fmt, variant, idf)
        x_dev = embedding_matrix(dev, vectors, fmt, variant, idf)
        notes["vector coverage"] = f"{vectors.coverage([as_tokens(r, fmt, variant) for r in dev]):.1%}"

    else:                                                    # M12 similarity numbers
        x_train = feature_matrix(train, vectors=vectors)
        x_dev = feature_matrix(dev, vectors=vectors)
        scaler = StandardScaler().fit(x_train)               # so one big number cannot dominate
        x_train, x_dev = scaler.transform(x_train), scaler.transform(x_dev)

    model = make_classifier(learner, seed)

    # A learner that ran out of iterations has not finished fitting, so its score is not
    # trustworthy. Raw word counts do this to the SVM (values up to 55, no upper bound),
    # while TF-IDF does not (everything scaled to 1). Say so instead of quietly reporting
    # the number - it is a real result about why scaling matters.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model.fit(x_train, [r["label"] for r in train])
    if any(issubclass(w.category, ConvergenceWarning) for w in caught):
        notes["WARNING"] = "did not converge - treat this score as unreliable"

    predicted = [int(p) for p in model.predict(x_dev)]

    if representation == "similarity" and hasattr(model, "coef_"):
        ranked = sorted(zip(feature_names(vectors), model.coef_[0]),
                        key=lambda kv: -abs(kv[1]))[:3]
        notes["leans on"] = ", ".join(f"{n} {w:+.2f}" for n, w in ranked)

    return predicted, confidence(model, x_dev), notes


# =============================================================================
# PART 3 - RUNNING AND RECORDING
# =============================================================================

def log_run(name: str, seed: int, fmt: str, variant: str, result, target: float,
            notes: dict[str, str], note: str = "") -> None:
    """Append one row to the experiment log. A run that is not logged did not happen."""
    family = MODELS[name][0]
    existing = {row[0] for row in csv.reader(LOG_FILE.open(encoding="utf-8")) if row}
    run_id = next(f"{family.lower()}_{name}_{fmt}_{variant}_s{seed}_{i}"
                  for i in range(1, 99)
                  if f"{family.lower()}_{name}_{fmt}_{variant}_s{seed}_{i}" not in existing)
    detail = "; ".join(f"{k} {v}" for k, v in notes.items())
    if note:
        detail = f"{note}; {detail}"
    with LOG_FILE.open("a", encoding="utf-8", newline="") as fh:
        csv.writer(fh).writerow([
            run_id, date.today().isoformat(), f"{family}_{name}", fmt, f"text_bn_{variant}",
            seed, "", "", "", "dev_usable", f"{result.macro_f1:.3f}", "",
            f"has-context+hard {target:.3f}; {detail}"])


def run_one(name: str, seed: int, fmt: str, variant: str, vectors: SkipGram | None,
            quiet: bool = False) -> tuple[evaluate.Result, float, dict[str, str]]:
    """Train one model, print its full report, and hand back the headline numbers."""
    train, dev = load_split("train"), load_split("dev")
    started = time.time()
    predicted, scores, notes = train_and_predict(name, train, dev, seed, fmt, variant, vectors)
    notes["trained in"] = f"{time.time() - started:.1f}s"

    target_rows = [i for i, r in enumerate(dev) if r["context"] and r["difficulty"] == "hard"]
    target = evaluate.macro_f1([dev[i]["label"] for i in target_rows],
                               [predicted[i] for i in target_rows])

    if quiet:
        result = evaluate.evaluate(dev, predicted, scores)
    else:
        label = f"{MODELS[name][0]} {name}  ({fmt}, {variant}, seed {seed})"
        result = evaluate.report(dev, predicted, scores, name=label)
        print("  " + " | ".join(f"{k}: {v}" for k, v in notes.items()))
    return result, target, notes


def run_all(seeds: tuple[int, ...], fmt: str, variant: str, write_log: bool,
            note: str = "") -> int:
    """Train every model and put them side by side, weakest to strongest."""
    rows = []
    for name in MODELS:
        model_seeds = seeds if needs_a_seed(name) else (seeds[0],)
        scores, targets, notes = [], [], {}
        for seed in model_seeds:
            seed_vectors = load_or_train_vectors(seed) if uses_vectors(name) else None
            result, target, notes = run_one(name, seed, fmt, variant, seed_vectors, quiet=True)
            scores.append(result.macro_f1)
            targets.append(target)
            if write_log:
                log_run(name, seed, fmt, variant, result, target, notes, note)
        mean, spread = evaluate.summarise_seeds(scores)
        rows.append((name, MODELS[name][0], mean, spread, float(np.mean(targets)),
                     len(model_seeds), dict(notes)))
        flag = "  <- " + notes["WARNING"] if "WARNING" in notes else ""
        print(f"  done: {name:<24} dev {mean:.3f}  ({notes.get('trained in', '')}){flag}")

    rows.sort(key=lambda row: row[2])
    print("\n" + "=" * 84)
    print(f"THE CLASSICAL MODELS on dev  ({fmt}, {variant})")
    print("=" * 84)
    print(f"  {'model':<24}{'lab':>5}{'dev macro-F1':>14}{'has-ctx + hard':>16}{'seeds':>7}")
    print(f"  {'-' * 74}")
    for name, family, mean, spread, target, n_seeds, model_notes in rows:
        spread_text = f" ±{spread:.3f}" if n_seeds > 1 else "       "
        flag = "  (did not converge)" if "WARNING" in model_notes else ""
        print(f"  {name:<24}{family:>5}{mean:>9.3f}{spread_text}{target:>16.3f}{n_seeds:>7}{flag}")

    print(f"\n  The bars they have to clear, on the same dev records:")
    dev = load_split("dev")
    grounded = [r for r in dev if r["context"]]
    hard = [r for r in grounded if r["difficulty"] == "hard"]
    for level, rows_ in (("has-context", grounded), ("has-context + hard", hard)):
        bars = evaluate.baselines_for(rows_)
        print(f"    {level:<20}" + "   ".join(f"{k} {v:.3f}" for k, v in bars.items()))
    print("\n  A model beating 'exact string match' overall but not on hard has learned the")
    print("  shortcut, not the task. The hard column is the one that counts (PRD G4).")
    print("=" * 84)
    return 0


def needs_a_seed(name: str) -> bool:
    """True when the model has randomness in it, so it must run on all three seeds."""
    return MODELS[name][2] == "xgb" or MODELS[name][1].startswith("skipgram")


def uses_vectors(name: str) -> bool:
    """True when the model needs the Skip-gram word vectors."""
    return MODELS[name][1].startswith("skipgram") or MODELS[name][1] == "similarity"


def load_or_train_vectors(seed: int = 42) -> SkipGram:
    """The word vectors for one seed, trained once and then reused.

    Skip-gram starts from random numbers, so seed 42 and seed 1337 give different
    vectors. Reusing one set for all three seeds would hide that variation and make the
    spread look smaller than it is, so each seed gets its own file.
    """
    path = VECTORS_FILE.with_name(f"skipgram_s{seed}.npz")
    if path.exists():
        return SkipGram.load(path)
    print(f"  training word vectors for seed {seed} (about 3 minutes)")
    model = train_on_split(seed=seed, verbose=False)
    model.save(path)
    return model


# =============================================================================
# PART 4 - OUR CODE AGAINST THE LIBRARY'S (guide §7.1)
# =============================================================================

def lab_naive_bayes(texts: list[str], labels: list[int]) -> dict:
    """Lab 3's Naive Bayes, written the lab's way: count, smooth, work in logs."""
    import collections
    import math
    vocabulary = {w for t in texts for w in t.split()}
    counts = {0: collections.Counter(), 1: collections.Counter()}
    documents = collections.Counter(labels)
    for text, label in zip(texts, labels):
        counts[label].update(text.split())
    model = {"prior": {}, "likelihood": {}, "vocabulary": vocabulary}
    for label in (0, 1):
        model["prior"][label] = math.log(documents[label] / len(labels))
        total = sum(counts[label].values())
        model["likelihood"][label] = {
            word: math.log((counts[label][word] + 1) / (total + len(vocabulary)))
            for word in vocabulary}
    return model


def lab_naive_bayes_predict(model: dict, text: str) -> int:
    scores = {}
    for label in (0, 1):
        scores[label] = model["prior"][label] + sum(
            model["likelihood"][label][w] for w in text.split() if w in model["vocabulary"])
    return max(scores, key=scores.get)


def run_lab_check() -> int:
    """Does the library agree with the lab's own code? The guide asks for this on 200 records."""
    train, dev = load_split("train")[:600], load_split("dev")[:200]
    train_texts = [" ".join(as_tokens(r, "F1")) for r in train]
    dev_texts = [" ".join(as_tokens(r, "F1")) for r in dev]
    labels = [r["label"] for r in train]

    ours = lab_naive_bayes(train_texts, labels)
    our_predictions = [lab_naive_bayes_predict(ours, t) for t in dev_texts]

    vectorizer = CountVectorizer(lowercase=False, token_pattern=r"\S+")
    library = MultinomialNB(alpha=1.0).fit(vectorizer.fit_transform(train_texts), labels)
    library_predictions = [int(p) for p in library.predict(vectorizer.transform(dev_texts))]

    agreement = sum(a == b for a, b in zip(our_predictions, library_predictions)) / len(dev_texts)
    print("=" * 78)
    print("THE LAB'S CODE vs THE LIBRARY (Naive Bayes, 600 train / 200 dev records)")
    print("=" * 78)
    print(f"  they predict the same label on {agreement:.1%} of the 200 dev records")
    print(f"  the lab's version scores  {evaluate.macro_f1([r['label'] for r in dev], our_predictions):.3f}")
    print(f"  the library's version     {evaluate.macro_f1([r['label'] for r in dev], library_predictions):.3f}")
    print("\n  Same maths: count each word per class, add 1 so nothing is impossible, then")
    print("  add up the logs. The library is used elsewhere only because it is faster.")
    print("=" * 78)
    return 0 if agreement > 0.95 else 1


# =============================================================================
# COMMAND LINE
# =============================================================================

def main() -> int:
    parser = argparse.ArgumentParser(description="The classical models (Labs 2-3; M1, M2, M11, M12).")
    parser.add_argument("--model", choices=list(MODELS), help="train one model")
    parser.add_argument("--all", action="store_true", help="train every model, side by side")
    parser.add_argument("--lab-check", action="store_true", help="our code vs the library's")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", action="store_true", help="use all three project seeds")
    parser.add_argument("--format", default="F2", choices=["F1", "F2", "F3"])
    parser.add_argument("--variant", default="V1",
                        choices=["V0", "V1", "V2", "V3", "V4", "V2-demo"])
    parser.add_argument("--log", action="store_true", help="append to the experiment log")
    parser.add_argument("--note", default="", help="a note to store with the logged run")
    args = parser.parse_args()

    seeds = evaluate.SEEDS if args.seeds else (args.seed,)

    if args.lab_check:
        return run_lab_check()
    if args.all:
        return run_all(seeds, args.format, args.variant, args.log, args.note)
    if args.model:
        scores = []
        for seed in (seeds if needs_a_seed(args.model) else (seeds[0],)):
            vectors = load_or_train_vectors(seed) if uses_vectors(args.model) else None
            result, target, notes = run_one(args.model, seed, args.format, args.variant, vectors)
            scores.append(result.macro_f1)
            if args.log:
                log_run(args.model, seed, args.format, args.variant, result, target,
                        notes, args.note)
        if len(scores) > 1:
            mean, spread = evaluate.summarise_seeds(scores)
            print(f"\n  across {len(scores)} seeds: {mean:.3f} ± {spread:.3f}")
        return 0
    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
