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
from dataclasses import dataclass, field
from datetime import date

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
from text_bn import preprocess as tokenise_variant
from skipgram import SkipGram, inverse_document_frequency, train_on_split, vectors_path
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

# The experiment log lives in evaluate.py and is reached through evaluate.ensure_log_file().
# No copy of the path is kept here on purpose: a module-level alias is bound at import time
# and would go stale the moment the real one changed.


# =============================================================================
# PART 1 - TURNING RECORDS INTO NUMBERS
# =============================================================================

def parts_as_text(records: list[dict], fmt: str, variant: str) -> dict[str, list[str]]:
    """The records' three pieces of text, ready for counting.

    F1 leaves the passage out, so its block is simply not built.

    Each part is tokenised on its own. It used to be routed through as_tokens(), which is
    built for ONE flat sequence and therefore inserts a "<SEP>" between the pieces - so
    every passage column ended "... <SEP> <SEP>" and every question column "... <SEP>".
    Those markers belong in a flat sequence, not in a column that holds a single part.
    """
    wanted = ["question", "answer"] if fmt == "F1" else ["context", "question", "answer"]
    columns: dict[str, list[str]] = {name: [] for name in wanted}
    for record in records:
        parts = record_parts(record, raw=(variant == "V0"))
        text_of = {"context": parts.context, "question": parts.question, "answer": parts.answer}
        for name in wanted:
            columns[name].append(" ".join(tokenise_variant(text_of[name], variant)))
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


@dataclass
class FittedModel:
    """One trained model plus everything needed to turn a NEW record into its numbers.

    Training and serving must not be two different code paths. If the web demo rebuilt the
    feature pipeline on its own, the two would drift apart the moment either changed, and
    the demo would quietly show predictions from a model that is not the one in Table 5.
    So the fitting lives here once, and both `train_and_predict` (which scores dev) and
    `src/serving.py` (which answers one typed-in record) go through it.

    Everything here was fitted on TRAIN ONLY, which is what makes it safe to apply to
    anything else.
    """

    name: str
    representation: str
    fmt: str
    variant: str
    seed: int
    classifier: object | None = None            # None for the language-model family
    features: SparseFeatures | None = None      # word/character counting
    scaler: StandardScaler | None = None        # M12 only
    idf: dict[str, float] | None = None         # skipgram_tfidf only
    style: AnswerStyleClassifier | None = None  # M11b only
    vectors: SkipGram | None = None
    notes: dict[str, str] = field(default_factory=dict)

    def matrix(self, records: list[dict]):
        """The records as numbers, using the pieces fitted on train."""
        if self.representation in ("bow", "tfidf"):
            return self.features.transform(parts_as_text(records, self.fmt, self.variant))
        if self.representation.startswith("skipgram"):
            return embedding_matrix(records, self.vectors, self.fmt, self.variant, self.idf)
        x = feature_matrix(records, vectors=self.vectors)    # M12 similarity numbers
        return self.scaler.transform(x)

    def predict(self, records: list[dict]) -> list[int]:
        if self.representation == "language_model":
            return [self.style.predict(r) for r in records]
        return [int(p) for p in self.classifier.predict(self.matrix(records))]

    def confidence(self, records: list[dict]) -> list[float] | None:
        if self.representation == "language_model":
            return [self.style.score_gap(r) for r in records]
        return confidence(self.classifier, self.matrix(records))


def fit_model(name: str, train: list[dict], seed: int, fmt: str, variant: str,
              vectors: SkipGram | None) -> FittedModel:
    """Train one model on the train split and hand back everything it needs to be used."""
    _family, representation, learner = MODELS[name]
    fitted = FittedModel(name=name, representation=representation, fmt=fmt, variant=variant,
                         seed=seed, vectors=vectors)

    if representation == "language_model":
        # M11(b): one character language model per class - it reads only the answer.
        fitted.style = AnswerStyleClassifier().fit(train)
        return fitted

    if representation in ("bow", "tfidf"):
        fitted.features = SparseFeatures(representation)
        x_train = fitted.features.fit_transform(parts_as_text(train, fmt, variant))
        fitted.notes["features"] = f"{fitted.features.size:,}"

    elif representation.startswith("skipgram"):
        assert vectors is not None
        if representation.endswith("tfidf"):
            fitted.idf = inverse_document_frequency([as_tokens(r, fmt, variant) for r in train])
        x_train = embedding_matrix(train, vectors, fmt, variant, fitted.idf)

    else:                                                    # M12 similarity numbers
        raw = feature_matrix(train, vectors=vectors)
        fitted.scaler = StandardScaler().fit(raw)            # so one big number cannot dominate
        x_train = fitted.scaler.transform(raw)

    fitted.classifier = make_classifier(learner, seed)

    # A learner that ran out of iterations has not finished fitting, so its score is not
    # trustworthy. Raw word counts do this to the SVM (values up to 55, no upper bound),
    # while TF-IDF does not (everything scaled to 1). Say so instead of quietly reporting
    # the number - it is a real result about why scaling matters.
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        fitted.classifier.fit(x_train, [r["label"] for r in train])
    if any(issubclass(w.category, ConvergenceWarning) for w in caught):
        fitted.notes["WARNING"] = "did not converge - treat this score as unreliable"

    if representation == "similarity" and hasattr(fitted.classifier, "coef_"):
        ranked = sorted(zip(feature_names(vectors), fitted.classifier.coef_[0]),
                        key=lambda kv: -abs(kv[1]))[:3]
        fitted.notes["leans on"] = ", ".join(f"{n} {w:+.2f}" for n, w in ranked)

    return fitted


def train_and_predict(name: str, train: list[dict], dev: list[dict], seed: int,
                      fmt: str, variant: str, vectors: SkipGram | None
                      ) -> tuple[list[int], list[float] | None, dict[str, str]]:
    """Fit one model on train and predict dev. Returns (predictions, confidence, notes)."""
    fitted = fit_model(name, train, seed, fmt, variant, vectors)
    notes = dict(fitted.notes)

    # These two describe the DEV set against the fitted model, so they cannot be computed
    # at fitting time - a served model has no dev set.
    if fitted.representation in ("bow", "tfidf"):
        notes["dev OOV"] = f"{out_of_vocabulary_rate(parts_as_text(train, fmt, variant), parts_as_text(dev, fmt, variant)):.1%}"
    elif fitted.representation.startswith("skipgram"):
        notes["vector coverage"] = f"{vectors.coverage([as_tokens(r, fmt, variant) for r in dev]):.1%}"

    return fitted.predict(dev), fitted.confidence(dev), notes


# =============================================================================
# PART 3 - RUNNING AND RECORDING
# =============================================================================

def log_run(name: str, seed: int, fmt: str, variant: str, result, target: float,
            notes: dict[str, str], note: str = "") -> None:
    """Append one row to the experiment log. A run that is not logged did not happen."""
    family = MODELS[name][0]
    # One path object for both the read and the write. Reading a freshly-resolved path but
    # writing `LOG_FILE` (bound at import) is two sources of truth: if the log ever moves,
    # the run ids would be checked against one file and appended to another, which is how
    # duplicate ids get created.
    log = evaluate.ensure_log_file()
    existing = {row[0] for row in csv.reader(log.open(encoding="utf-8")) if row}
    stem = f"{family.lower()}_{name}_{fmt}_{variant}_s{seed}"
    attempt = 1
    while f"{stem}_{attempt}" in existing:        # no upper limit: a long-running project
        attempt += 1                              # should never crash just for logging
    run_id = f"{stem}_{attempt}"
    detail = "; ".join(f"{k} {v}" for k, v in notes.items())
    if note:
        detail = f"{note}; {detail}"
    with log.open("a", encoding="utf-8", newline="") as fh:
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
        warnings_seen = {}          # kept across seeds: see below
        for seed in model_seeds:
            seed_vectors = load_or_train_vectors(seed, variant) if uses_vectors(name) else None
            result, target, notes = run_one(name, seed, fmt, variant, seed_vectors, quiet=True)
            scores.append(result.macro_f1)
            targets.append(target)
            # `notes` is overwritten each seed, so a model that failed to converge on seed 42
            # but converged on seed 2024 would be reported as clean. Keep every warning.
            if "WARNING" in notes:
                warnings_seen[seed] = notes["WARNING"]
            if write_log:
                log_run(name, seed, fmt, variant, result, target, notes, note)
        mean, spread = evaluate.summarise_seeds(scores)
        summary = dict(notes)
        if warnings_seen:
            summary["WARNING"] = (f"{next(iter(warnings_seen.values()))} "
                                  f"(seeds {', '.join(str(s) for s in warnings_seen)})")
        rows.append((name, MODELS[name][0], mean, spread, float(np.mean(targets)),
                     len(model_seeds), summary))
        flag = "  <- " + summary["WARNING"] if "WARNING" in summary else ""
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


# =============================================================================
# PART 3b - M10: DOES THE CLASSIC TEXT CLEANUP ACTUALLY HELP? (guide §6)
# =============================================================================

# The six ways of preparing text, from guide §6. V1 is what everything else uses.
ABLATION_VARIANTS = ["V0", "V1", "V2", "V3", "V4", "V2-demo"]

# One model per family, chosen by its score on the target (has-context + hard):
# tfidf_logreg was the joint-best M1 at 0.610, skipgram_mean_xgb the best M2 at 0.529.
# M3 (the recurrent models) joins this table once it exists.
ABLATION_MODELS = ["tfidf_logreg", "skipgram_mean_xgb"]

VARIANT_MEANING = {
    "V0": "no cleaning at all, just split on spaces",
    "V1": "clean + tokenize (the default)",
    "V2": "V1 + drop common words (negation kept)",
    "V3": "V1 + stem",
    "V4": "V1 + drop common words + stem",
    "V2-demo": "V2 but negation and numbers dropped too",
}


def run_ablation(seed: int, fmt: str, write_log: bool, write_table: bool) -> int:
    """Train each model once per variant and lay the scores side by side.

    One seed, not three: the M2 row needs its own word vectors for every variant (three
    minutes each), and this experiment is about the gap between variants, not about
    squeezing the last thousandth out of any one of them.
    """
    train, dev = load_split("train"), load_split("dev")
    target_rows = [i for i, r in enumerate(dev) if r["context"] and r["difficulty"] == "hard"]
    table = []

    for name in ABLATION_MODELS:
        for variant in ABLATION_VARIANTS:
            vectors = load_or_train_vectors(seed, variant) if uses_vectors(name) else None
            started = time.time()
            predicted, _, notes = train_and_predict(name, train, dev, seed, fmt, variant, vectors)

            overall = evaluate.macro_f1([r["label"] for r in dev], predicted)
            target = evaluate.macro_f1([dev[i]["label"] for i in target_rows],
                                       [predicted[i] for i in target_rows])
            by_type = {kind: score for kind, _, _, score in evaluate.by_error_type(dev, predicted)}
            row = {"model": name, "variant": variant, "overall": round(overall, 3),
                   "has_context_hard": round(target, 3),
                   "contradiction": round(by_type.get("contradiction", float("nan")), 3),
                   "numeric": round(by_type.get("numeric", float("nan")), 3),
                   "relational": round(by_type.get("relational", float("nan")), 3),
                   "features": notes.get("features", notes.get("vector coverage", "")),
                   "seconds": round(time.time() - started, 1)}
            table.append(row)
            # flush: this loop takes the better part of an hour, and without it Python holds
            # the output in a buffer until the very end, so a working run looks like a hung one.
            print(f"  done: {name:<22} {variant:<8} overall {overall:.3f}  hard {target:.3f}"
                  f"  contradiction {row['contradiction']:.3f}", flush=True)
            if write_log:
                result = evaluate.evaluate(dev, predicted)
                log_run(name, seed, fmt, variant, result, target, notes, "M10 ablation")

    print("\n" + "=" * 92)
    print(f"M10 - DOES THE CLASSIC CLEANUP HELP?  (dev, {fmt}, seed {seed})")
    print("=" * 92)
    for name in ABLATION_MODELS:
        rows = [r for r in table if r["model"] == name]
        best = max(r["overall"] for r in rows)
        print(f"\n  {name}")
        print(f"  {'variant':<10}{'overall':>9}{'hard':>8}{'contradiction':>15}{'numeric':>9}"
              f"{'relational':>12}   what it does")
        for row in rows:
            mark = " *" if row["overall"] == best else "  "
            print(f"  {row['variant']:<10}{row['overall']:>9.3f}{row['has_context_hard']:>8.3f}"
                  f"{row['contradiction']:>15.3f}{row['numeric']:>9.3f}{row['relational']:>12.3f}"
                  f"{mark} {VARIANT_MEANING[row['variant']]}")

    print("\n  How to read this:")
    print("    * marks the best overall score for that model.")
    print("    V2-demo is the warning shot: it throws away না/নয় and the number words, so if")
    print("    the 'contradiction' column drops there, that is the damage this project's")
    print("    protected-word list exists to prevent.")
    print("=" * 92)

    if write_table:
        path = evaluate.write_table(table, "table6_preprocessing_ablation.csv")
        print(f"  wrote {path}")
    return 0


def needs_a_seed(name: str) -> bool:
    """True when the model has randomness in it, so it must run on all three seeds."""
    return MODELS[name][2] == "xgb" or MODELS[name][1].startswith("skipgram")


def uses_vectors(name: str) -> bool:
    """True when the model needs the Skip-gram word vectors."""
    return MODELS[name][1].startswith("skipgram") or MODELS[name][1] == "similarity"


def load_or_train_vectors(seed: int = 42, variant: str = "V1") -> SkipGram:
    """The word vectors for one seed and one preprocessing variant, cached on disk.

    Two reasons this is keyed by BOTH:
      * Skip-gram starts from random numbers, so seed 42 and seed 1337 give different
        vectors. Sharing one set across seeds would hide that variation.
      * The words themselves change with the variant. Stemming turns "কলেজের" into
        "কলেজ", so vectors learned on unstemmed text would not recognise most of the
        stemmed ones. The M10 experiment would then be comparing nothing at all.
    """
    path = vectors_path(seed, variant)          # skipgram.py owns the naming, so it cannot drift
    if path.exists():
        return SkipGram.load(path)
    print(f"  training word vectors for {variant}, seed {seed} (about 3 minutes)", flush=True)
    model = train_on_split(seed=seed, verbose=False, variant=variant)
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
    parser.add_argument("--ablation", action="store_true",
                        help="M10: every preprocessing variant, side by side")
    parser.add_argument("--table", action="store_true", help="write results/tables/*.csv")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", action="store_true", help="use all three project seeds")
    # F1 and F2 only. F3's entire meaning is "the passage is a separate segment", which a
    # bag of words and a flat token list cannot represent - as_tokens(F3) is byte-identical
    # to as_tokens(F2). Accepting it here would let a run be logged as input_format=F3 while
    # actually being F2, and the M9 format comparison would then report "no difference"
    # without ever having tried F3. F3 belongs to the pretrained encoders (M4/M5).
    parser.add_argument("--format", default="F2", choices=["F1", "F2"],
                        help="F1 = question + answer, F2 = passage + question + answer. "
                             "F3 is encoder-only (see src/preprocess.py).")
    parser.add_argument("--variant", default="V1",
                        choices=["V0", "V1", "V2", "V3", "V4", "V2-demo"])
    parser.add_argument("--log", action="store_true", help="append to the experiment log")
    parser.add_argument("--note", default="", help="a note to store with the logged run")
    args = parser.parse_args()

    seeds = evaluate.SEEDS if args.seeds else (args.seed,)

    if args.lab_check:
        return run_lab_check()
    if args.ablation:
        return run_ablation(args.seed, args.format, args.log, args.table)
    if args.all:
        return run_all(seeds, args.format, args.variant, args.log, args.note)
    if args.model:
        scores = []
        for seed in (seeds if needs_a_seed(args.model) else (seeds[0],)):
            vectors = load_or_train_vectors(seed, args.variant) if uses_vectors(args.model) else None
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
