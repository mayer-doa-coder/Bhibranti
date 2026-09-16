"""Corpus validity audit — the Phase 1 release gate (PRD section 5.5, guide section 9).

Runs four probes plus the structural checks. Exits non-zero if a **blocking**
gate fails, so this can sit in front of training in a pipeline.

LABEL CONVENTION (PRD section 5.1a): ``1 = correct/faithful``, ``0 = incorrect/
hallucinated``.

    python src/audit.py --data data/splits
    python src/audit.py --data data/interim/bn_pool.jsonl --probe metadata

Probes
------
metadata     PRD V1, **blocking**. Logistic regression on content-free surface
             features of the candidate answer alone. Must score < 0.60 macro-F1.
answer-only  PRD V3. Character n-grams over the candidate answer alone. Should
             sit well below a full-input model on has-context items.
overlap      Not in the PRD. Measures whether a faithful answer can be spotted
             purely by string-matching it against the context. A high score here
             means the has-context split is an extraction task, not a
             hallucination-detection task.
prior        Majority-class and per-question-prior sanity baselines.

Every probe uses a **grouped** split so that a question and its answers never
straddle train and test; an ungrouped split flatters every probe.
"""

from __future__ import annotations

import argparse
import collections
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupShuffleSplit

SEED = 42

METADATA_FEATURES = [
    "n_tokens",
    "n_chars",
    "mean_token_len",
    "latin_ratio",
    "comma",
    "period",
    "question_mark",
    "exclamation",
    "quote",
    "digit_ratio",
    "caps_ratio",
]

# PRD section 5.5 / guide section 9.1
GATE_BLOCKING = 0.60
GATE_CLEAN = 0.55


def load(path: Path) -> list[dict]:
    files = sorted(path.glob("*.jsonl")) if path.is_dir() else [path]
    rows: list[dict] = []
    for f in files:
        with f.open(encoding="utf-8") as fh:
            rows.extend(json.loads(line) for line in fh if line.strip())
    if not rows:
        raise SystemExit(f"no records found under {path}")
    return rows


def metadata_features(text: str) -> list[float]:
    """Surface features containing no content words whatsoever."""
    text = str(text)
    tokens = text.split()
    n = max(len(tokens), 1)
    return [
        len(tokens),
        len(text),
        float(np.mean([len(t) for t in tokens])) if tokens else 0.0,
        sum(c.isascii() and c.isalpha() for c in text) / max(len(text), 1),
        text.count(","),
        text.count("."),
        text.count("?"),
        text.count("!"),
        text.count('"'),
        sum(c.isdigit() for c in text) / max(len(text), 1),
        sum(1 for t in tokens if t.isupper()) / n,
    ]


def grouped_split(rows: list[dict], test_size: float = 0.2):
    groups = np.array([r.get("pair_id", r["question"]) for r in rows])
    y = np.array([r["label"] for r in rows])
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=SEED)
    return next(splitter.split(np.zeros(len(rows)), y, groups))


def verdict(score: float) -> str:
    if score < GATE_CLEAN:
        return "CLEAN"
    if score < GATE_BLOCKING:
        return "BORDERLINE"
    return "ARTIFACTED"


def probe_metadata(rows: list[dict]) -> float:
    X = np.array([metadata_features(r["candidate_answer"]) for r in rows])
    y = np.array([r["label"] for r in rows])
    tr, te = grouped_split(rows)
    clf = LogisticRegression(max_iter=2000).fit(X[tr], y[tr])
    score = f1_score(y[te], clf.predict(X[te]), average="macro")
    top = sorted(zip(METADATA_FEATURES, clf.coef_[0]), key=lambda kv: -abs(kv[1]))[:4]
    print(f"  metadata-only macro-F1      = {score:.3f}   [{verdict(score)}]")
    print("    leading coefficients:", ", ".join(f"{n}={c:+.3f}" for n, c in top))
    return score


def probe_answer_only(rows: list[dict]) -> float:
    texts = [str(r["candidate_answer"]) for r in rows]
    y = np.array([r["label"] for r in rows])
    tr, te = grouped_split(rows)
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=2)
    A = vec.fit_transform([texts[i] for i in tr])
    B = vec.transform([texts[i] for i in te])
    clf = LogisticRegression(max_iter=1000).fit(A, y[tr])
    score = f1_score(y[te], clf.predict(B), average="macro")
    print(f"  answer-only char-ngram F1   = {score:.3f}")
    return score


def probe_overlap(rows: list[dict]) -> float | None:
    grounded = [r for r in rows if r["condition"] == "has_context" and r["context"]]
    if not grounded:
        print("  overlap probe               = n/a (no has-context records)")
        return None
    y = np.array([r["label"] for r in grounded])
    # answer copied from context -> predict correct (1); see PRD section 5.1a
    predictions = np.array(
        [
            1 if str(r["candidate_answer"]).strip().rstrip("।. ") in r["context"] else 0
            for r in grounded
        ]
    )
    score = f1_score(y, predictions, average="macro")
    print(f"  context-substring rule F1   = {score:.3f}   (has-context only, n={len(grounded):,})")
    if score >= 0.70:
        print("    WARNING: faithful answers are largely copied verbatim from the context.")
        print("    A string matcher solves this split without detecting hallucination.")
    return score


def probe_prior(rows: list[dict]) -> None:
    y = [r["label"] for r in rows]
    majority = collections.Counter(y).most_common(1)[0][0]
    score = f1_score(y, [majority] * len(y), average="macro")
    print(f"  majority-class baseline F1  = {score:.3f}")


def structural_checks(rows: list[dict]) -> list[str]:
    problems: list[str] = []
    labels = collections.Counter(r["label"] for r in rows)
    n = len(rows)

    hallucinated = labels[0] / n
    if not 0.45 <= hallucinated <= 0.55:
        problems.append(f"class balance {hallucinated:.1%} hallucinated, outside 50%+/-5%")

    conditions = collections.Counter(r["condition"] for r in rows)
    has_ctx = conditions["has_context"] / n
    if not 0.55 <= has_ctx <= 0.65:
        problems.append(f"has-context share {has_ctx:.1%}, outside the 60%+/-5% target")

    ids = collections.Counter(r["id"] for r in rows)
    if dupes := [i for i, c in ids.items() if c > 1]:
        problems.append(f"{len(dupes)} duplicate ids")

    # PRD section 5.1a: hallucination_type == "none" exactly when label == 1.
    for r in rows:
        if r["label"] == 1 and r["hallucination_type"] != "none":
            problems.append(f"{r['id']}: label=1 (correct) but hallucination_type={r['hallucination_type']}")
            break
        if r["label"] == 0 and r["hallucination_type"] == "none":
            problems.append(f"{r['id']}: label=0 (hallucinated) but hallucination_type=none")
            break

    unlabeled = sum(1 for r in rows if r.get("hallucination_type") == "unlabeled")
    if unlabeled:
        problems.append(
            f"{unlabeled:,} records carry provisional hallucination_type='unlabeled' "
            "(not human-annotated; PRD D8 unmet)"
        )

    # Phase 1 is Bengali script only (PRD section 7.1); Banglish is Phase 2.
    not_bengali = sum(1 for r in rows if r.get("script_condition") != "bengali_script")
    if not_bengali:
        problems.append(
            f"{not_bengali:,} records are not script_condition='bengali_script' — "
            "Phase 1 is Bengali only (PRD section 7.1)"
        )

    annotated = sum(1 for r in rows if r.get("annotator_1") is not None)
    if annotated == 0:
        problems.append("no human annotation present (PRD D4/D5 unmet)")

    return problems


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True, help="jsonl file or directory")
    parser.add_argument(
        "--probe",
        choices=["all", "metadata", "answer-only", "overlap"],
        default="all",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="also fail on non-blocking structural problems",
    )
    args = parser.parse_args()

    rows = load(args.data)
    print("=" * 68)
    print(f"CORPUS AUDIT — {args.data}  ({len(rows):,} records)")
    print("=" * 68)

    print("\nPROBES (grouped split, no pair spans train/test)")
    metadata_score = None
    if args.probe in ("all", "metadata"):
        metadata_score = probe_metadata(rows)
    if args.probe in ("all", "answer-only"):
        probe_answer_only(rows)
    if args.probe in ("all", "overlap"):
        probe_overlap(rows)
    if args.probe == "all":
        probe_prior(rows)

    print("\nSTRUCTURAL CHECKS")
    problems = structural_checks(rows)
    if problems:
        for p in problems:
            print(f"  [!] {p}")
    else:
        print("  all structural checks passed")

    print("\n" + "=" * 68)
    failed = False
    if metadata_score is not None:
        if metadata_score >= GATE_BLOCKING:
            print(f"GATE V1 FAILED: metadata probe {metadata_score:.3f} >= {GATE_BLOCKING}")
            print("Fix the generation prompt and regenerate. Do not adjust this threshold.")
            failed = True
        else:
            print(f"GATE V1 PASSED: metadata probe {metadata_score:.3f} < {GATE_BLOCKING}")
    if args.strict and problems:
        print(f"STRICT MODE: {len(problems)} structural problem(s) treated as failure")
        failed = True
    print("=" * 68)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
