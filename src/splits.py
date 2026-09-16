"""Load the Phase 1 splits the way every training and evaluation script must.

    from splits import load_split
    train = load_split("train")          # excluded pairs already removed
    dev = load_split("dev")

PRD Q5 / §5.1d: pairs that human review found unusable -- a stored label
confirmed wrong, or a broken item -- are left out of BOTH training and scoring.
`src/merge_annotation.py` marks them `excluded = true`; this is the one place
that filters them, so no script can apply the decision differently.

`include_excluded=True` exists only for audits and for reporting what was
removed. Never train or score with it.
"""

from __future__ import annotations

import json
from pathlib import Path

SPLITS = Path("data/splits")


def load_split(split: str, include_excluded: bool = False) -> list[dict]:
    if split not in ("train", "dev", "test"):
        raise ValueError(f"unknown split {split!r}")
    path = SPLITS / f"{split}.jsonl"
    rows = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
    if any("excluded" not in r for r in rows):
        raise SystemExit(f"{path} has no `excluded` field - run src/merge_annotation.py first "
                         f"(a corpus rebuild resets it)")
    if include_excluded:
        return rows
    return [r for r in rows if not r["excluded"]]
