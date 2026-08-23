"""Build the 100-item blind annotation-agreement test (PRD M2).

Picks 100 items out of data/corpus/bn_v1/corpus.jsonl, taking AT MOST ONE
member per pair_id. Both members of a pair are the correct and the hallucinated
answer to the SAME question -- if both appeared in the test an annotator could
work out the labels by comparing them instead of applying the guidelines, and
the agreement score would look good without the guidelines having been tested.

Everything here is Bangla script. Phase 1 has no Banglish.

Produces two files that must stay separate:

  items_for_annotation_BLANK.csv   no label shown. Copy it TWICE, one per
                                   annotator; they fill it in independently.
  answer_key_DO_NOT_OPEN_YET.csv   the real labels. Opened only AFTER both
                                   annotators submit. Opening it early makes
                                   the test measure memory, not the guidelines.

Run from the repository root:

    python src/build_agreement_test.py
"""

from __future__ import annotations

import collections
import csv
import json
import random
from pathlib import Path

CORPUS = Path("data/corpus/bn_v1/corpus.jsonl")
OUT_DIR = Path("data/annotated/agreement_test_v1")

SEED = 42

# (condition, label, difficulty) -> how many items.
# Mirrors the corpus: 60/40 has-context, 50/50 label, and roughly the corpus's
# own share of hard pairs (33% has-context, 21% no-context) so the resulting
# kappa describes the corpus you will actually annotate, not an easier or
# harder slice of it.
QUOTA = {
    ("has_context", 1, "hard"): 10, ("has_context", 1, "easy"): 20,
    ("has_context", 0, "hard"): 10, ("has_context", 0, "easy"): 20,
    ("no_context", 1, "hard"): 4,  ("no_context", 1, "easy"): 16,
    ("no_context", 0, "hard"): 4,  ("no_context", 0, "easy"): 16,
}


def main() -> None:
    rng = random.Random(SEED)
    records = [json.loads(l) for l in CORPUS.open(encoding="utf-8") if l.strip()]

    used_pairs: set[str] = set()
    selected: list[dict] = []
    for (cond, label, diff), n in sorted(QUOTA.items()):
        pool = [r for r in records
                if r["condition"] == cond and r["label"] == label
                and r["difficulty"] == diff and r["pair_id"] not in used_pairs]
        rng.shuffle(pool)
        if len(pool) < n:
            raise ValueError(f"only {len(pool)} available for {cond}/{label}/{diff}, need {n}")
        chosen = pool[:n]
        used_pairs.update(r["pair_id"] for r in chosen)
        selected += chosen

    assert len(selected) == 100, f"expected 100 items, got {len(selected)}"
    pair_ids = [r["pair_id"] for r in selected]
    assert len(pair_ids) == len(set(pair_ids)), "a pair_id was used twice -- sibling leakage"

    # Shuffle so items are not grouped by condition/label/difficulty -- the
    # grouping alone would let an annotator infer the pattern.
    rng.shuffle(selected)
    for i, r in enumerate(selected):
        r["_item_id"] = f"item_{i + 1:03d}"

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    blank_fields = ["item_id", "condition", "context", "question", "answer",
                    "your_label", "your_type", "your_difficulty", "your_notes"]
    with (OUT_DIR / "items_for_annotation_BLANK.csv").open(
            "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=blank_fields)
        w.writeheader()
        for r in selected:
            w.writerow({
                "item_id": r["_item_id"],
                "condition": r["condition"],
                "context": "" if r["condition"] == "no_context" else r["context"],
                "question": r["question"],
                "answer": r["candidate_answer"],
                "your_label": "", "your_type": "",
                "your_difficulty": "", "your_notes": "",
            })

    key_fields = ["item_id", "condition", "difficulty", "subject",
                  "true_label", "true_label_word", "source_pair_id"]
    with (OUT_DIR / "answer_key_DO_NOT_OPEN_YET.csv").open(
            "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=key_fields)
        w.writeheader()
        for r in selected:
            w.writerow({
                "item_id": r["_item_id"],
                "condition": r["condition"],
                "difficulty": r["difficulty"],
                "subject": r["subject"],
                "true_label": r["label"],
                "true_label_word": "correct" if r["label"] == 1 else "wrong",
                "source_pair_id": r["pair_id"],
            })

    print(f"wrote 100 items to {OUT_DIR}/")
    c = collections.Counter((r["condition"], r["label"], r["difficulty"]) for r in selected)
    for k in sorted(c):
        print(f"  {k[0]:<12} label={k[1]} ({'correct' if k[1] else 'wrong  '}) "
              f"{k[2]:<5}: {c[k]}")
    print(f"\nunique source pairs: {len(set(pair_ids))}/100 "
          f"(sibling leakage: {'NONE' if len(set(pair_ids)) == 100 else 'YES'})")


if __name__ == "__main__":
    main()
