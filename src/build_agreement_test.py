"""Build the 100-item blind annotation-agreement test (PRD M2, guide sec 5.1).

Picks 100 records out of the 500-pair pilot (data/generated/pilot_v1/pilot.jsonl),
choosing AT MOST ONE member per pair_id so no two items in the set are the
correct/wrong answers to the same question -- that would let an annotator
guess by comparing, instead of judging each item on its own, which would
inflate the agreement score without the guidelines actually being tested.

Produces two separate files that must stay separate:

  items_for_annotation_BLANK.csv   -- no label shown. Copy this file TWICE,
                                       one per annotator. They fill it in
                                       independently, without seeing each
                                       other's copy.
  answer_key_DO_NOT_OPEN_YET.csv   -- the real label/type/difficulty. Only
                                       opened AFTER both annotators submit,
                                       to score them. Opening it early defeats
                                       the whole point of the test.

Run from the repository root:

    python src/build_agreement_test.py
"""

from __future__ import annotations

import collections
import csv
import json
import random
from pathlib import Path

PILOT = Path("data/generated/pilot_v1/pilot.jsonl")
OUT_DIR = Path("data/annotated/agreement_test_v1")

SEED = 42
# Mirrors the pilot's own 60/40 has-context/no-context split and the
# corpus-wide 50/50 label balance (PRD D2, D3), scaled down to 100 items.
N_HAS_CONTEXT_CORRECT = 30
N_HAS_CONTEXT_WRONG = 30
N_NO_CONTEXT_CORRECT = 20
N_NO_CONTEXT_WRONG = 20


def load_records() -> list[dict]:
    return [json.loads(l) for l in PILOT.open(encoding="utf-8") if l.strip()]


def pick_one_per_pair(records: list[dict], condition: str, label: int, n: int, rng: random.Random,
                       used_pairs: set[str]) -> list[dict]:
    """Picks n records matching condition+label, each from a pair_id not
    already used elsewhere in the test set (see module docstring)."""
    pool = [r for r in records if r["condition"] == condition and r["label"] == label
            and r["pair_id"] not in used_pairs]
    rng.shuffle(pool)
    if len(pool) < n:
        raise ValueError(f"only {len(pool)} candidates for condition={condition} label={label}, need {n}")
    chosen = pool[:n]
    used_pairs.update(r["pair_id"] for r in chosen)
    return chosen


def main() -> None:
    rng = random.Random(SEED)
    records = load_records()
    used_pairs: set[str] = set()

    selected: list[dict] = []
    selected += pick_one_per_pair(records, "has_context", 1, N_HAS_CONTEXT_CORRECT, rng, used_pairs)
    selected += pick_one_per_pair(records, "has_context", 0, N_HAS_CONTEXT_WRONG, rng, used_pairs)
    selected += pick_one_per_pair(records, "no_context", 1, N_NO_CONTEXT_CORRECT, rng, used_pairs)
    selected += pick_one_per_pair(records, "no_context", 0, N_NO_CONTEXT_WRONG, rng, used_pairs)

    # Validation before writing anything.
    assert len(selected) == 100, f"expected 100 items, got {len(selected)}"
    pair_ids = [r["pair_id"] for r in selected]
    assert len(pair_ids) == len(set(pair_ids)), "a pair_id was used twice -- sibling leakage"
    item_ids_check = [r["id"] for r in selected]
    assert len(item_ids_check) == len(set(item_ids_check)), "duplicate source record picked twice"

    # Shuffle presentation order so items aren't grouped by condition/label --
    # that grouping alone would let an annotator guess the pattern.
    rng.shuffle(selected)

    # Assign short, neutral item ids that reveal nothing about the label.
    for i, r in enumerate(selected):
        r["_item_id"] = f"item_{i+1:03d}"

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    blank_fields = ["item_id", "condition", "context", "question", "answer",
                     "your_label", "your_type", "your_difficulty", "your_notes"]
    with (OUT_DIR / "items_for_annotation_BLANK.csv").open("w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=blank_fields)
        w.writeheader()
        for r in selected:
            w.writerow({
                "item_id": r["_item_id"],
                "condition": r["condition"],
                "context": r["context"],
                "question": r["question"],
                "answer": r["candidate_answer"],
                "your_label": "",
                "your_type": "",
                "your_difficulty": "",
                "your_notes": "",
            })

    key_fields = ["item_id", "condition", "true_label", "true_label_word",
                   "true_type_guess", "source_pair_id", "bn_question", "bn_answer"]
    with (OUT_DIR / "answer_key_DO_NOT_OPEN_YET.csv").open("w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=key_fields)
        w.writeheader()
        for r in selected:
            w.writerow({
                "item_id": r["_item_id"],
                "condition": r["condition"],
                "true_label": r["label"],
                "true_label_word": "correct" if r["label"] == 1 else "wrong",
                "true_type_guess": r["hallucination_type"],
                "source_pair_id": r["pair_id"],
                "bn_question": r["bn_question"],
                "bn_answer": r["bn_candidate_answer"],
            })

    log_fields = ["item_id", "condition", "label", "subject", "source_pair_id"]
    with (OUT_DIR / "selection_log.csv").open("w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=log_fields)
        w.writeheader()
        for r in selected:
            w.writerow({
                "item_id": r["_item_id"],
                "condition": r["condition"],
                "label": r["label"],
                "subject": r["subject"],
                "source_pair_id": r["pair_id"],
            })

    counts = collections.Counter((r["condition"], r["label"]) for r in selected)
    print(f"wrote 100 items to {OUT_DIR}/")
    print("breakdown:")
    for (cond, lab), n in sorted(counts.items()):
        print(f"  {cond:<12} label={lab} ({'correct' if lab==1 else 'wrong'}): {n}")
    print(f"\nunique source pairs used: {len(set(pair_ids))} (must equal 100: {len(set(pair_ids))==100})")


if __name__ == "__main__":
    main()
