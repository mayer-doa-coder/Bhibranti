"""Trained models, kept on disk and ready to answer one record at a time.

Training a model and serving it are two different jobs with two different shapes. Training
reads the whole train split and scores the whole dev split. Serving gets ONE record, typed
in by a person, and has to answer in milliseconds. This module is the bridge.

    python src/serving.py --build        # train every model once and save it (~6 minutes)
    python src/serving.py --check        # load them back and predict one record

WHY THE MODELS ARE SAVED RATHER THAN RETRAINED
Retraining twelve models takes about six minutes. A web page cannot make someone wait that
long, and retraining per request would also mean the demo showed a slightly different model
each time. So they are fitted once, written to disk, and loaded read-only.

THE ONE RULE THIS FILE EXISTS TO KEEP
Serving uses `train_classical.fit_model()` - the same function that produced the numbers in
Table 5. It does not rebuild the feature pipeline on its own. If it did, the demo and the
results table would drift apart the first time either changed, and the demo would be showing
predictions from a model nobody ever measured.

WHAT IS *NOT* HERE
No test-split anything. The demo trains on train, quotes dev scores, and never opens
`data/splits/test.jsonl` (PRD: it is opened exactly once, at M6).
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import joblib

sys.path.insert(0, str(Path(__file__).resolve().parent))

import evaluate
from features import FUZZY_THRESHOLD, extract_features
from splits import load_split
from train_classical import (
    MODELS,
    FittedModel,
    fit_model,
    load_or_train_vectors,
    uses_vectors,
)

# =============================================================================
# SETTINGS
# =============================================================================

STORE = Path(__file__).resolve().parents[1] / "data" / "processed" / "serving"

# The demo serves the settings everything else was measured under, so the scores on the
# page are the scores in Table 5 - not a different configuration that happens to look better.
SERVING_SEED = 42
SERVING_FORMAT = "F2"
SERVING_VARIANT = "V1"

# Which models the demo offers. All twelve, so the comparison page is the real ladder and
# not a flattering subset. bow_svm is included precisely BECAUSE it is broken - see below.
SERVING_MODELS = list(MODELS)

# Plain-language names, for a page read by someone who has not seen the code.
MODEL_TITLE = {
    "bow_nb": "Bag of Words + Naive Bayes",
    "tfidf_nb": "TF-IDF + Naive Bayes",
    "bow_logreg": "Bag of Words + Logistic Regression",
    "tfidf_logreg": "TF-IDF + Logistic Regression",
    "bow_svm": "Bag of Words + linear SVM",
    "tfidf_svm": "TF-IDF + linear SVM",
    "char_lm": "Character language model (answer only)",
    "skipgram_mean_logreg": "Skip-gram averaged + Logistic Regression",
    "skipgram_tfidf_logreg": "Skip-gram, IDF-weighted + Logistic Regression",
    "skipgram_mean_xgb": "Skip-gram averaged + XGBoost",
    "skipgram_tfidf_xgb": "Skip-gram, IDF-weighted + XGBoost",
    "similarity_logreg": "Similarity numbers + Logistic Regression",
}

# What each one can and cannot see. This is the honest part of the demo: several of these
# models physically cannot check the answer against the passage, and the page says so.
#
# Kept to roughly 45 characters so each one stays on a single line under the model name in
# the table. Longer strings wrap and make the rows uneven.
MODEL_SEES = {
    "bow_nb": "word counts, kept separate for each part",
    "tfidf_nb": "the same, common words down-weighted",
    "bow_logreg": "word counts, kept separate for each part",
    "tfidf_logreg": "the same, common words down-weighted",
    "bow_svm": "word counts, kept separate for each part",
    "tfidf_svm": "the same, common words down-weighted",
    "char_lm": "the answer text only, nothing else",
    "skipgram_mean_logreg": "average word meaning, word order lost",
    "skipgram_tfidf_logreg": "the same, rare words weighted higher",
    "skipgram_mean_xgb": "average word meaning, word order lost",
    "skipgram_tfidf_xgb": "the same, rare words weighted higher",
    "similarity_logreg": "how closely the answer matches the passage",
}


# =============================================================================
# PART 1 - BUILDING THE STORE
# =============================================================================

def store_path(name: str) -> Path:
    return STORE / f"{name}_s{SERVING_SEED}_{SERVING_FORMAT}_{SERVING_VARIANT}.joblib"


def build(only: list[str] | None = None, verbose: bool = True) -> list[str]:
    """Fit every model on the train split and write it to disk. Returns what it wrote."""
    train = load_split("train")
    STORE.mkdir(parents=True, exist_ok=True)
    written = []
    for name in (only or SERVING_MODELS):
        started = time.time()
        vectors = (load_or_train_vectors(SERVING_SEED, SERVING_VARIANT)
                   if uses_vectors(name) else None)
        fitted = fit_model(name, train, SERVING_SEED, SERVING_FORMAT, SERVING_VARIANT, vectors)
        joblib.dump(fitted, store_path(name), compress=3)
        written.append(name)
        if verbose:
            size = store_path(name).stat().st_size / 1e6
            flag = "  (does not converge)" if "WARNING" in fitted.notes else ""
            print(f"  built {name:<24} {time.time() - started:6.1f}s  {size:5.1f} MB{flag}",
                  flush=True)
    return written


def is_built() -> bool:
    return all(store_path(name).exists() for name in SERVING_MODELS)


def missing() -> list[str]:
    return [name for name in SERVING_MODELS if not store_path(name).exists()]


# =============================================================================
# PART 2 - USING THEM
# =============================================================================

def make_record(question: str, answer: str, passage: str = "") -> dict:
    """A typed-in question/answer/passage, shaped like a corpus record.

    Only three fields are real input. The rest are the shape the feature code expects, and
    `label` is deliberately absent: nothing in the serving path may read a label, because a
    record typed in by a person does not have one.
    """
    return {
        "question": question.strip(),
        "candidate_answer": answer.strip(),
        "context": passage.strip(),
        "condition": "has_context" if passage.strip() else "no_context",
        "difficulty": "unknown",
        "subject": "unknown",
        "pair_id": "typed_in",
        "id": "typed_in",
    }


@dataclass
class Prediction:
    model: str
    title: str
    sees: str
    verdict: int                 # 1 = faithful/correct, 0 = hallucinated
    confidence: float | None     # raw score; direction is "higher = more likely correct"
    scale: str                   # what that number IS - see below
    milliseconds: float
    unreliable: bool = False

    @property
    def label(self) -> str:
        return "correct" if self.verdict == 1 else "hallucinated"

    @property
    def confidence_text(self) -> str:
        """The confidence, written so it cannot be misread.

        These numbers are not the same kind of thing. Logistic Regression, Naive Bayes and
        XGBoost give a probability between 0 and 1. A linear SVM gives a signed distance
        from its dividing line, which has no upper bound and is not a probability. The
        character language model gives the gap between two likelihood scores. Printing all
        three as "87%" would be inventing a certainty that two of them never expressed.
        """
        if self.confidence is None:
            return "—"
        if self.scale == "probability":
            return f"{self.confidence:.0%} sure correct"
        if self.scale == "margin":
            # Kept to one short line so the table row does not wrap. It still says "distance"
            # rather than a percentage, because an SVM has no probability to report.
            side = "correct" if self.confidence >= 0 else "made up"
            return f"leans {side} by {abs(self.confidence):.2f}"
        # Three decimals, not two: these gaps are routinely smaller than 0.01, and at two
        # decimals every one of them prints as "+0.00", which says nothing at all.
        return f"score gap {self.confidence:+.3f}"


class Ensemble:
    """Every saved model, loaded once, ready to be asked the same question."""

    def __init__(self, models: dict[str, FittedModel]) -> None:
        self.models = models

    @classmethod
    def load(cls, names: list[str] | None = None) -> Ensemble:
        loaded = {}
        for name in (names or SERVING_MODELS):
            path = store_path(name)
            if path.exists():
                loaded[name] = joblib.load(path)
        if not loaded:
            raise FileNotFoundError(
                "No trained models found. Run `python src/serving.py --build` first.")
        return cls(loaded)

    def predict_one(self, record: dict) -> list[Prediction]:
        """Ask every model about one record, timing each separately."""
        out = []
        for name, fitted in self.models.items():
            started = time.perf_counter()
            verdict = fitted.predict([record])[0]
            scores = fitted.confidence([record])
            elapsed = (time.perf_counter() - started) * 1000
            out.append(Prediction(
                model=name,
                title=MODEL_TITLE.get(name, name),
                sees=MODEL_SEES.get(name, ""),
                verdict=int(verdict),
                confidence=float(scores[0]) if scores is not None else None,
                scale=confidence_scale(fitted),
                milliseconds=elapsed,
                unreliable="WARNING" in fitted.notes,
            ))
        return out


def confidence_scale(fitted: FittedModel) -> str:
    """Which KIND of number this model's confidence is. See Prediction.confidence_text."""
    if fitted.representation == "language_model":
        return "gap"
    if hasattr(fitted.classifier, "predict_proba"):
        return "probability"
    return "margin"


# =============================================================================
# PART 3 - THE EVIDENCE BEHIND A DECISION
# =============================================================================

# Plain-language readings of the similarity features, so the page can explain a decision
# instead of only announcing it.
# Kept short on purpose: these are table rows in a narrow column, not sentences to read.
EVIDENCE_ORDER = [
    ("exact_in_passage", "In the passage, word for word",
     lambda v: "yes" if v >= 0.5 else "no"),
    ("token_overlap", "Answer words found in passage", lambda v: f"{v:.0%}"),
    ("edit_passage", "Letters that must change", lambda v: f"{v:.0%}"),
    ("lm_passage", "Matches passage style", lambda v: f"{v:.2f}"),
    ("edit_question", "Repeats the question", lambda v: f"{1 - v:.0%}"),
    ("answer_len", "Answer length", lambda v: f"{int(v)} chars"),
    ("digit_share", "Digits in the answer", lambda v: f"{v:.0%}"),
]


def evidence(record: dict) -> list[dict]:
    """The M12 similarity numbers for one record, in readable form.

    These are the same numbers `similarity_logreg` is given, so this is not a separate
    explanation invented for the page - it is literally what that model sees.
    """
    values = extract_features(record)
    rows = []
    for key, description, render in EVIDENCE_ORDER:
        if key not in values:
            continue
        if not record["context"] and key in ("exact_in_passage", "token_overlap",
                                             "edit_passage", "lm_passage"):
            rows.append({"key": key, "what": description, "value": "no passage",
                         "raw": None, "missing": True})
            continue
        rows.append({"key": key, "what": description, "value": render(values[key]),
                     "raw": round(float(values[key]), 4), "missing": False})
    return rows


def rule_verdicts(record: dict) -> list[dict]:
    """The two no-learning baselines, run on the same record.

    Any model that cannot beat these has not earned its place, so the page shows them next
    to the models rather than in a footnote.
    """
    # `rule` is the name used in the experiment log and the guide; `short` is what the page
    # shows. Both live here rather than in the template, so the wording is testable and the
    # page keeps no vocabulary of its own.
    values = extract_features(record)
    if not record["context"]:
        return [{"rule": "Exact string match", "short": "Answer is in the passage",
                 "verdict": None, "why": "needs a passage; this record has none"},
                {"rule": "Fuzzy string match (V6)", "short": "Answer is nearly in the passage",
                 "verdict": None, "why": "needs a passage; this record has none"}]
    exact = int(values["exact_in_passage"] >= 0.5)
    fuzzy = int(values["edit_passage"] <= FUZZY_THRESHOLD)
    return [
        {"rule": "Exact string match", "short": "Answer is in the passage", "verdict": exact,
         "why": ("the answer appears word for word in the passage" if exact
                 else "the answer does not appear word for word in the passage")},
        {"rule": "Fuzzy string match (V6)", "short": "Answer is nearly in the passage",
         "verdict": fuzzy,
         "why": (f"at most {FUZZY_THRESHOLD:.0%} of the answer's letters must change to find "
                 f"it in the passage" if fuzzy else
                 f"more than {FUZZY_THRESHOLD:.0%} of the answer's letters would have to change")},
    ]


# =============================================================================
# PART 4 - THE SCOREBOARD
# =============================================================================

# Measured at step 11.5 on the dev split, F2 / V1, and recorded in results/experiment_log.csv.
# Read from the log rather than hard-coded, so the page can never disagree with the log.
def scoreboard() -> list[dict]:
    """Each model's dev scores, straight out of the experiment log.

    Averaged over seeds, not "the last row wins". A model with randomness in it was run on
    all three seeds, and Table 5 of the guide reports the mean. If this page took only the
    final row it would show skipgram_mean_xgb at 0.489 (seed 2024) where the guide says
    0.503 (the mean of 0.519, 0.501, 0.489) - the page and the write-up would disagree, and
    whichever the reader saw second would look like a mistake.
    """
    per_seed: dict[str, dict[int, dict]] = {}
    log = evaluate.ensure_log_file()
    for row in csv.DictReader(log.open(encoding="utf-8")):
        if row["model"].startswith("NOTE") or "M10 ablation" in row["notes"]:
            continue                      # ablation rows vary by variant; not the headline run
        if row["preprocessing"] != f"text_bn_{SERVING_VARIANT}" or row["input_format"] != SERVING_FORMAT:
            continue
        name = row["model"].split("_", 1)[-1] if "_" in row["model"] else row["model"]
        if name not in MODELS or not row["dev_macro_f1"]:
            continue
        target = None
        for piece in row["notes"].split(";"):
            if "has-context+hard" in piece:
                target = float(piece.split()[-1])
        # Later rows for the same seed replace earlier ones: a rerun supersedes its original.
        per_seed.setdefault(name, {})[int(row["seed"])] = {
            "overall": float(row["dev_macro_f1"]),
            "hard": target,
            "unreliable": "did not converge" in row["notes"],
            "run_id": row["run_id"],
        }

    rows = []
    for name, seeds in per_seed.items():
        runs = list(seeds.values())
        overall, spread = evaluate.summarise_seeds([r["overall"] for r in runs])
        hard_values = [r["hard"] for r in runs if r["hard"] is not None]
        rows.append({
            "model": name,
            "title": MODEL_TITLE.get(name, name),
            "sees": MODEL_SEES.get(name, ""),
            "overall": overall,
            "spread": spread,
            "hard": sum(hard_values) / len(hard_values) if hard_values else None,
            "seeds": len(runs),
            "unreliable": any(r["unreliable"] for r in runs),
            "run_id": runs[-1]["run_id"],
        })

    ranked = sorted(rows, key=lambda r: -(r["hard"] or 0))
    for position, row in enumerate(ranked, start=1):
        row["rank"] = position
    return ranked


# The bars every model has to clear, on the dev hard subset. Measured, not assumed.
BASELINES = [
    {"name": "Exact string match", "hard": 0.487,
     "what": "does the answer appear word for word in the passage?"},
    {"name": "Fuzzy string match (V6)", "hard": 0.591,
     "what": "does it NEARLY appear, allowing ~40% of letters to differ?"},
    {"name": "Always say 'correct'", "hard": 0.333,
     "what": "no model at all - the floor"},
]


def main() -> int:
    parser = argparse.ArgumentParser(description="Train the demo's models and save them.")
    parser.add_argument("--build", action="store_true", help="fit every model and write it to disk")
    parser.add_argument("--rebuild", action="store_true", help="rebuild even if already present")
    parser.add_argument("--check", action="store_true", help="load them back and try one record")
    parser.add_argument("--model", help="build or check just this one")
    args = parser.parse_args()

    if args.build or args.rebuild:
        wanted = [args.model] if args.model else SERVING_MODELS
        todo = wanted if args.rebuild else [n for n in wanted if not store_path(n).exists()]
        if not todo:
            print(f"  all {len(wanted)} models already built in {STORE}")
        else:
            print(f"  building {len(todo)} model(s) into {STORE}")
            build(todo)
        return 0

    if args.check:
        if not is_built():
            print(f"  not built yet: {', '.join(missing())}")
            print("  run: python src/serving.py --build")
            return 1
        ensemble = Ensemble.load([args.model] if args.model else None)
        record = make_record(
            question="বাংলাদেশের রাজধানী কোথায়?",
            answer="ঢাকা",
            passage="বাংলাদেশের রাজধানী ঢাকা। এটি দেশের বৃহত্তম শহর।")
        print(f"\n  question: {record['question']}")
        print(f"  answer  : {record['candidate_answer']}\n")
        for p in sorted(ensemble.predict_one(record), key=lambda p: p.model):
            mark = " (unreliable)" if p.unreliable else ""
            print(f"    {p.title:<44} {p.label:<14} {p.milliseconds:6.1f} ms{mark}")
        print("\n  evidence:")
        for row in evidence(record):
            print(f"    {row['what']:<58} {row['value']}")
        return 0

    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
