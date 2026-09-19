"""Scoring: the one place that turns predictions into the numbers we report (guide §12).

Every model on the ladder is judged here, so no model can be scored on its own terms.
It exists before the first model for exactly that reason.

    from evaluate import evaluate, report, macro_f1

    result = evaluate(records, predictions)          # one slice
    report(records, predictions, name="TF-IDF + LogReg")   # the full breakdown

Run from the repository root:

    python src/evaluate.py --baselines           # score the rule baselines on dev
    python src/evaluate.py --baselines --table   # ...and write results/tables/
    python src/evaluate.py --demo                # what a made-up perfect/lazy model scores

WHAT GETS REPORTED, AND WHY EACH SPLIT IS SEPARATE
  with passage / without passage  two different tasks; one average hides both
  easy / hard                     hard is the number this project is judged on
  by subject                      law and science need outside knowledge, so a collapse
                                  there is a finding, not a defect
  by error type                   which kind of mistake the model misses
A single blended number is never reported on its own (PRD §5.4).

THREE THINGS THIS FILE IS CAREFUL ABOUT
  * Slice sizes are printed, and anything under SMALL_SLICE records is marked "small" -
    dev has subjects with only 22 records, where a score means very little.
  * The confidence interval resamples PAIRS, not records. The two answers to one question
    share a passage, so treating them as independent would make the interval look tighter
    than it really is.
  * The string-matching baselines are recomputed on exactly the records being scored,
    never copied from another table.
"""

from __future__ import annotations

import argparse
import collections
import csv
import math
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from text_bn import clean, tokenize

# =============================================================================
# SETTINGS
# =============================================================================

# A slice smaller than this is reported, but marked - dev has subjects with 22 records,
# where one flipped answer moves macro-F1 by several points.
SMALL_SLICE = 60

# Resamples for the confidence interval (guide §12.2).
BOOTSTRAP_SAMPLES = 1000

# The project's three seeds. Anything with randomness runs on all three.
SEEDS = (42, 1337, 2024)

TABLES_DIR = Path(__file__).resolve().parents[1] / "results" / "tables"

# The experiment log. Every script that runs a model appends to this one file, so it lives
# here - the module they all already import - rather than being spelled out in each of them.
LOG_FILE = Path(__file__).resolve().parents[1] / "results" / "experiment_log.csv"
LOG_HEADER = ["run_id", "date", "model", "input_format", "preprocessing", "seed", "lr",
              "batch", "epochs", "split", "dev_macro_f1", "test_macro_f1", "notes"]


def ensure_log_file() -> Path:
    """Create the log with its header row if it is not there, and hand back the path.

    Call this before reading or appending. Without it, a fresh clone that has not committed
    `results/` - or anyone who deleted the folder - loses a finished training run to a
    FileNotFoundError at the very last step, after the model has already been trained.
    """
    if not LOG_FILE.exists():
        LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
        with LOG_FILE.open("w", encoding="utf-8", newline="") as fh:
            csv.writer(fh).writerow(LOG_HEADER)
    return LOG_FILE


# =============================================================================
# PART 1 - THE BASIC MEASURES, written out rather than imported
# =============================================================================

def confusion(truth: list[int], predicted: list[int], label: int) -> tuple[int, int, int]:
    """Counts for one class: (found correctly, claimed wrongly, missed)."""
    hit = sum(t == label and p == label for t, p in zip(truth, predicted))
    false_alarm = sum(t != label and p == label for t, p in zip(truth, predicted))
    missed = sum(t == label and p != label for t, p in zip(truth, predicted))
    return hit, false_alarm, missed


def precision_recall_f1(truth: list[int], predicted: list[int], label: int) -> tuple[float, float, float]:
    """For one class: how often it was right, how much of it it found, and the balance."""
    hit, false_alarm, missed = confusion(truth, predicted, label)
    precision = hit / (hit + false_alarm) if hit + false_alarm else 0.0
    recall = hit / (hit + missed) if hit + missed else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def macro_f1(truth: list[int], predicted: list[int]) -> float:
    """The project's main number: the F1 of each class, averaged.

    Averaging the two is what stops a model from winning by always answering "correct" -
    that would score 0 on the other class and so about 0.33 overall.
    """
    return sum(precision_recall_f1(truth, predicted, label)[2] for label in (0, 1)) / 2


def accuracy(truth: list[int], predicted: list[int]) -> float:
    return sum(t == p for t, p in zip(truth, predicted)) / len(truth) if truth else 0.0


def auc(truth: list[int], scores: list[float] | None) -> float | None:
    """Chance the model gives a random correct answer a higher score than a wrong one.

    Needs the model's confidence, not just its yes/no answer, so it is None without one.
    """
    if scores is None or len(set(truth)) < 2:
        return None
    order = sorted(range(len(scores)), key=lambda i: scores[i])
    ranks = [0.0] * len(scores)
    i = 0
    while i < len(order):                      # average the ranks of any tied scores
        j = i
        while j + 1 < len(order) and scores[order[j + 1]] == scores[order[i]]:
            j += 1
        for k in range(i, j + 1):
            ranks[order[k]] = (i + j) / 2 + 1
        i = j + 1
    positives = sum(truth)
    negatives = len(truth) - positives
    rank_sum = sum(r for r, t in zip(ranks, truth) if t == 1)
    return (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


# =============================================================================
# PART 2 - ONE SLICE'S RESULT
# =============================================================================

@dataclass
class Result:
    """Everything worth knowing about one group of records."""
    name: str
    n: int
    macro_f1: float
    accuracy: float
    correct_class: tuple[float, float, float]        # precision, recall, F1 for label 1
    wrong_class: tuple[float, float, float]          # precision, recall, F1 for label 0
    auc: float | None = None
    counts: dict[str, int] = field(default_factory=dict)

    @property
    def is_small(self) -> bool:
        return self.n < SMALL_SLICE

    def line(self) -> str:
        auc_text = f"{self.auc:.3f}" if self.auc is not None else "  -  "
        mark = " (small)" if self.is_small else ""
        return (f"  {self.name:<24}{self.n:>7,}{self.macro_f1:>10.3f}{self.accuracy:>10.3f}"
                f"{self.wrong_class[1]:>10.3f}{auc_text:>9}{mark}")


def evaluate(records: list[dict], predicted: list[int], scores: list[float] | None = None,
             name: str = "all") -> Result:
    """Score one group of records. `scores` is the model's confidence, if it has any."""
    truth = [r["label"] for r in records]
    hit, false_alarm, missed = confusion(truth, predicted, 0)
    return Result(
        name=name,
        n=len(records),
        macro_f1=macro_f1(truth, predicted),
        accuracy=accuracy(truth, predicted),
        correct_class=precision_recall_f1(truth, predicted, 1),
        wrong_class=precision_recall_f1(truth, predicted, 0),
        auc=auc(truth, scores),
        counts={"hallucinations caught": hit, "false alarms": false_alarm,
                "hallucinations missed": missed},
    )


# =============================================================================
# PART 3 - THE SLICES (PRD §5.4)
# =============================================================================

def slice_by(records: list[dict], predicted: list[int], scores: list[float] | None,
             key: str) -> list[Result]:
    """Score each group of a field (condition, difficulty, subject), biggest group first."""
    groups: dict[str, list[int]] = collections.defaultdict(list)
    for i, record in enumerate(records):
        groups[record[key]].append(i)
    results = []
    for value, rows in groups.items():
        results.append(evaluate([records[i] for i in rows], [predicted[i] for i in rows],
                                [scores[i] for i in rows] if scores else None, name=value))
    return sorted(results, key=lambda r: r.n, reverse=True)


def by_error_type(records: list[dict], predicted: list[int]) -> list[tuple[str, int, float, float]]:
    """How the model does on each kind of mistake: (type, n wrong answers, caught, pair score).

    Two numbers, because one on its own would mislead:
      * "caught" is the share of THAT type's wrong answers the model flagged. A type slice
        holds only wrong answers, so macro-F1 on it alone would be meaningless.
      * "pair score" takes both answers of those questions, so both classes are present and
        the number can be compared with every other macro-F1 in the report.

    Records still marked `unlabeled` are left out - guide §12.1 - and counted separately.
    """
    by_pair: dict[str, list[int]] = collections.defaultdict(list)
    for i, record in enumerate(records):
        by_pair[record["pair_id"]].append(i)

    groups: dict[str, list[str]] = collections.defaultdict(list)
    for pair_id, rows in by_pair.items():
        wrong = [i for i in rows if records[i]["label"] == 0]
        if not wrong:
            continue
        kind = records[wrong[0]]["hallucination_type"]
        groups[kind].append(pair_id)

    out = []
    for kind, pair_ids in groups.items():
        if kind == "unlabeled":
            continue
        rows = [i for pid in pair_ids for i in by_pair[pid]]
        wrong_rows = [i for i in rows if records[i]["label"] == 0]
        caught = sum(predicted[i] == 0 for i in wrong_rows) / len(wrong_rows)
        pair_score = macro_f1([records[i]["label"] for i in rows], [predicted[i] for i in rows])
        out.append((kind, len(wrong_rows), caught, pair_score))
    return sorted(out, key=lambda row: row[1], reverse=True)


# =============================================================================
# PART 4 - THE BASELINES, RECOMPUTED ON THESE EXACT RECORDS
# =============================================================================

def baseline_predictions(records: list[dict], rule: str = "exact",
                         threshold: float | None = None) -> list[int]:
    """What a rule with no learning would answer on these records.

    rule="exact"  the answer appears in the passage, word for word
    rule="fuzzy"  the answer NEARLY appears (V6). The cut-off comes from features.py,
                  where it was tuned - it is never repeated here, so re-tuning cannot
                  leave this file quoting a stale number.
    Records with no passage always come out "wrong", which is all those rules can do.
    """
    # Imported inside the function: features.py imports macro_f1 from this file, so a
    # top-level import here would be circular.
    from features import FUZZY_THRESHOLD, extract_features

    if threshold is None:
        threshold = FUZZY_THRESHOLD

    predictions = []
    for record in records:
        features = extract_features(record)
        if rule == "exact":
            predictions.append(int(features["exact_in_passage"]))
        else:
            predictions.append(1 if features["edit_passage"] <= threshold else 0)
    return predictions


def baselines_for(records: list[dict]) -> dict[str, float]:
    """The rule scores on exactly these records, so a model can be compared fairly."""
    truth = [r["label"] for r in records]
    grounded = [i for i, r in enumerate(records) if r["context"]]
    if not grounded:
        return {}
    rows = [records[i] for i in grounded]
    truth = [truth[i] for i in grounded]
    return {
        "exact string match": macro_f1(truth, baseline_predictions(rows, "exact")),
        "fuzzy string match": macro_f1(truth, baseline_predictions(rows, "fuzzy")),
        "always says correct": macro_f1(truth, [1] * len(truth)),
    }


# =============================================================================
# PART 5 - IS THE DIFFERENCE REAL? (guide §12.2)
# =============================================================================

def bootstrap_interval(records: list[dict], predicted: list[int],
                       samples: int = BOOTSTRAP_SAMPLES, seed: int = 42) -> tuple[float, float]:
    """A 95% confidence interval for macro-F1, by resampling PAIRS.

    Resampling records one by one would be wrong here: the two answers to a question share
    a passage and a question, so they rise and fall together. Drawing whole pairs keeps
    that dependence and gives an honest, slightly wider interval.
    """
    by_pair: dict[str, list[int]] = collections.defaultdict(list)
    for i, record in enumerate(records):
        by_pair[record["pair_id"]].append(i)
    pair_ids = list(by_pair)

    rng = random.Random(seed)
    scores = []
    for _ in range(samples):
        drawn = [by_pair[rng.choice(pair_ids)] for _ in range(len(pair_ids))]
        rows = [i for group in drawn for i in group]
        scores.append(macro_f1([records[i]["label"] for i in rows], [predicted[i] for i in rows]))
    scores.sort()
    return scores[int(0.025 * samples)], scores[int(0.975 * samples)]


def mcnemar(truth: list[int], predicted_a: list[int], predicted_b: list[int]) -> tuple[int, int, float]:
    """Do two models really differ? Returns (A-only wins, B-only wins, p-value).

    Only the records where the two disagree carry any information: if A is right and B is
    wrong roughly as often as the reverse, the gap between their scores is noise.
    Written out here because `statsmodels` is not one of this project's dependencies.
    """
    a_only = sum(pa == t and pb != t for t, pa, pb in zip(truth, predicted_a, predicted_b))
    b_only = sum(pb == t and pa != t for t, pa, pb in zip(truth, predicted_a, predicted_b))
    n = a_only + b_only
    if n == 0:
        return 0, 0, 1.0
    # Exact two-sided sign test: how surprising is this split of n disagreements?
    tail = sum(math.comb(n, k) for k in range(min(a_only, b_only) + 1)) / 2 ** n
    return a_only, b_only, min(1.0, 2 * tail)


def summarise_seeds(scores: list[float]) -> tuple[float, float]:
    """Mean and spread across the three seeds - always reported together (PRD §5.4).

    The spread is the population standard deviation (numpy's default), not the sample one.
    With three seeds the two differ by a factor of 1.22, so it is worth being able to say
    which this is: it describes the spread of the three runs we actually did, and is not an
    estimate of how a fourth seed would land.
    """
    return float(np.mean(scores)), float(np.std(scores))


def tune_threshold(truth: list[int], scores: list[float]) -> tuple[float, float]:
    """Pick the cut-off that scores best. DEV ONLY - never on test (guide §10).

    A model's confidence is not calibrated, so 0.5 is rarely the best place to split.
    """
    best_threshold, best_score = 0.5, -1.0
    for threshold in np.arange(0.05, 0.96, 0.01):
        predicted = [1 if s >= threshold else 0 for s in scores]
        score = macro_f1(truth, predicted)
        if score > best_score:
            best_threshold, best_score = round(float(threshold), 2), score
    return best_threshold, best_score


# =============================================================================
# PART 6 - M14, THE WORD-ORDER TEST (guide §12.1a)
# =============================================================================

def shuffle_tokens(tokens: list[str], seed: int = 42) -> list[str]:
    """Same words, new order. For models that are handed a token list."""
    shuffled = list(tokens)
    random.Random(seed).shuffle(shuffled)
    return shuffled


def shuffled_record(record: dict, seed: int = 42) -> dict:
    """A copy of the record with the words inside each part jumbled.

    The passage, the question and the answer are shuffled separately, so the record still
    has its three pieces - only the order inside each is destroyed. Feed this back through
    the SAME trained model and compare the scores.

    A bag-of-words model must come out completely unchanged; if it moves, something in the
    pipeline is reading order when it should not be.
    """
    rng = random.Random(seed)
    jumbled = dict(record)
    for part in ("context", "question", "candidate_answer"):
        text = clean(record[part])
        if not text:
            continue
        words = tokenize(text)
        rng.shuffle(words)
        jumbled[part] = " ".join(words)
    return jumbled


# =============================================================================
# PART 7 - THE REPORT
# =============================================================================

def report(records: list[dict], predicted: list[int], scores: list[float] | None = None,
           name: str = "model", split: str = "dev", show_baselines: bool = True) -> Result:
    """Print the full breakdown for one model, and return its overall result."""
    overall = evaluate(records, predicted, scores, name="overall")
    low, high = bootstrap_interval(records, predicted)

    print("=" * 84)
    print(f"{name}  -  {split} split, {len(records):,} records "
          f"({len({r['pair_id'] for r in records}):,} pairs)")
    print("=" * 84)
    print(f"  macro-F1 {overall.macro_f1:.3f}   95% interval {low:.3f} to {high:.3f}   "
          f"accuracy {overall.accuracy:.3f}")
    print(f"  of the {sum(1 for r in records if r['label'] == 0):,} hallucinated answers: "
          f"{overall.counts['hallucinations caught']:,} caught, "
          f"{overall.counts['hallucinations missed']:,} missed, "
          f"{overall.counts['false alarms']:,} correct answers wrongly flagged")

    print(f"\n  {'slice':<24}{'n':>7}{'macro-F1':>10}{'accuracy':>10}{'caught':>10}{'AUC':>9}")
    print(f"  {'-' * 68}")
    for key in ("condition", "difficulty"):
        for result in slice_by(records, predicted, scores, key):
            print(result.line())

    # The project's headline target (PRD G4) is hard records THAT HAVE A PASSAGE. The plain
    # "hard" row above also counts hard no-context records, so it is a different number.
    target = [i for i, r in enumerate(records)
              if r["context"] and r["difficulty"] == "hard"]
    if target:
        result = evaluate([records[i] for i in target], [predicted[i] for i in target],
                          [scores[i] for i in target] if scores else None,
                          name="has-context + hard *")
        print(result.line())
        print("  * the target this project is judged on: macro-F1 >= 0.80 (PRD G4)")
    print(f"  {'-' * 68}")
    for result in slice_by(records, predicted, scores, "subject"):
        print(result.line())

    if show_baselines:
        # Positions, not record objects: two records can hold equal values, and comparing
        # dictionaries would be both slow and ambiguous.
        grounded = [i for i, r in enumerate(records) if r["context"]]
        hard = [i for i in grounded if records[i]["difficulty"] == "hard"]
        if grounded:
            print("\n  Rules with no learning, on these same records "
                  "(a model has to beat the best of them):")
            for level, rows in (("has-context", grounded), ("hard only", hard)):
                if not rows:
                    continue
                bars = baselines_for([records[i] for i in rows])
                mine = macro_f1([records[i]["label"] for i in rows], [predicted[i] for i in rows])
                bar_text = "   ".join(f"{k} {v:.3f}" for k, v in bars.items())
                verdict = "BEATS" if mine > max(bars.values()) else "does NOT beat"
                print(f"    {level:<12} this model {mine:.3f}  |  {bar_text}   -> {verdict}")

    types = by_error_type(records, predicted)
    if types:
        print(f"\n  By kind of mistake ({split} only). 'caught' is the share of that kind the")
        print("  model flagged; 'pair score' adds the matching correct answers so it is")
        print("  comparable with the macro-F1 numbers above.")
        print(f"  {'type':<24}{'n wrong':>9}{'caught':>9}{'pair score':>12}")
        for kind, n, caught, pair_score in types:
            mark = " (small)" if n < SMALL_SLICE else ""
            print(f"  {kind:<24}{n:>9,}{caught:>9.3f}{pair_score:>12.3f}{mark}")

    unlabelled = sum(1 for r in records if r["label"] == 0
                     and r["hallucination_type"] == "unlabeled")
    if unlabelled:
        print(f"  ({unlabelled:,} wrong answers have no type yet and are left out of that table.)")

    provenances = {r.get("provenance", "unknown") for r in records}
    if provenances == {"llm_generated"}:
        print("\n  No human-written vs generated split: every record in this corpus was built")
        print("  with LLM assistance, so that comparison cannot be made (data/SOURCES.md).")

    print("=" * 84)
    return overall


def write_table(rows: list[dict], filename: str) -> Path:
    """Save a results table as CSV, for the report.

    An empty table is refused rather than written. The column names come from the first row,
    so with no rows this used to die on `rows[0]` with a bare IndexError - at the very end of
    a long run, after the results were computed and with nothing in the message to say that
    an empty table was the cause.
    """
    if not rows:
        raise ValueError(f"refusing to write an empty table to {filename}: "
                         "there are no results, so the column names are unknown")
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    path = TABLES_DIR / filename
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return path


# =============================================================================
# COMMAND LINE
# =============================================================================

def run_baselines(split: str, write: bool) -> int:
    """Score the no-learning rules. This is Table 5, and it also proves the scoring works."""
    from splits import load_split

    records = load_split(split)
    rules = {
        "exact string match": baseline_predictions(records, "exact"),
        "fuzzy string match (V6)": baseline_predictions(records, "fuzzy"),
        "always says correct": [1] * len(records),
        "always says hallucinated": [0] * len(records),
    }

    table = []
    for name, predicted in rules.items():
        result = report(records, predicted, name=name, split=split, show_baselines=False)
        slices = {r.name: r.macro_f1 for r in slice_by(records, predicted, None, "condition")}
        slices.update({r.name: r.macro_f1 for r in slice_by(records, predicted, None, "difficulty")})
        table.append({"model": name, "n": len(records), "macro_f1": round(result.macro_f1, 3),
                      "accuracy": round(result.accuracy, 3),
                      **{k: round(v, 3) for k, v in slices.items()}})

    print("\nSide by side:")
    print(f"  {'rule':<28}{'overall':>9}{'has-ctx':>9}{'no-ctx':>9}{'easy':>9}{'hard':>9}")
    for row in table:
        print(f"  {row['model']:<28}{row['macro_f1']:>9.3f}{row.get('has_context', 0):>9.3f}"
              f"{row.get('no_context', 0):>9.3f}{row.get('easy', 0):>9.3f}{row.get('hard', 0):>9.3f}")

    truth = [r["label"] for r in records]
    a_only, b_only, p = mcnemar(truth, rules["exact string match"], rules["fuzzy string match (V6)"])
    print(f"\n  Are the exact and fuzzy rules really different? They disagree on "
          f"{a_only + b_only:,} records ({a_only:,} where exact wins, {b_only:,} where fuzzy does); "
          f"p = {p:.3g}")

    if write:
        path = write_table(table, f"table5_baselines_{split}.csv")
        print(f"  wrote {path}")
    return 0


def run_demo() -> int:
    """Show what the scoring says about three made-up models, so the numbers make sense."""
    from splits import load_split

    records = load_split("dev")
    truth = [r["label"] for r in records]
    rng = random.Random(42)

    pretenders = {
        "a perfect model": list(truth),
        "one that always says 'correct'": [1] * len(records),
        "one that guesses at random": [rng.randint(0, 1) for _ in records],
        "one that is right 80% of the time": [t if rng.random() < 0.8 else 1 - t for t in truth],
    }
    print(f"  {'made-up model':<36}{'macro-F1':>10}{'accuracy':>10}   what it shows")
    notes = ["the ceiling", "why macro-F1 is used: accuracy looks like 0.5, the score does not",
             "the floor", "a realistic-looking model"]
    for (name, predicted), note in zip(pretenders.items(), notes):
        result = evaluate(records, predicted)
        print(f"  {name:<36}{result.macro_f1:>10.3f}{result.accuracy:>10.3f}   {note}")

    print("\n  The word-order test (M14) jumbles the words inside each part:")
    record = next(r for r in records if len(tokenize(clean(r["question"]))) > 7)
    print(f"    before: {clean(record['question'])[:70]}")
    print(f"    after : {shuffled_record(record)['question'][:70]}")
    print("    Same words, no order left. A bag-of-words model must score exactly the")
    print("    same on both; a model that reads word order should get worse.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Scoring and result tables (guide §12).")
    parser.add_argument("--baselines", action="store_true", help="score the no-learning rules")
    parser.add_argument("--demo", action="store_true", help="what the numbers mean")
    parser.add_argument("--split", choices=["train", "dev", "test"], default="dev")
    parser.add_argument("--table", action="store_true", help="write results/tables/*.csv")
    parser.add_argument("--final", action="store_true",
                        help="required to touch test - the single M6 evaluation")
    args = parser.parse_args()

    if args.split == "test" and not args.final:
        print("Refusing: test.jsonl is scored exactly once, at the final evaluation (PRD §5.4,\n"
              "RK10). If this really is that moment, pass --final, and log the run.")
        return 1
    if args.split == "test":
        print("!" * 84)
        print("SCORING THE TEST SPLIT. This is meant to happen once, ever. Log it in\n"
              "results/experiment_log.csv, and do not tune anything afterwards.")
        print("!" * 84)

    if args.baselines:
        return run_baselines(args.split, args.table)
    if args.demo:
        return run_demo()
    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
