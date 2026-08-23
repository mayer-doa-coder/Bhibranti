"""Check filled annotation sheets before they are scored or merged (PRD M3).

Run it on a folder of sheets, or on single files:

    python src/validate_annotation.py --dir data/annotated/round1/test/tawhid
    python src/validate_annotation.py --dir data/annotated/round1/dev

It reports progress and every problem it can detect, and exits non-zero if
anything would corrupt a downstream score.

WHY THIS EXISTS
---------------
Every check below caught a real mistake in the first annotation round, or
guards a failure that round exposed:

  * A label was typed into the ANSWER cell instead of `your_label`, leaving
    the row apparently unlabelled and the answer text silently corrupted
    ("৩২ বর্গ একক।correct"). Comparing each row's question/answer against the
    corpus finds this instantly.
  * `your_type` and `your_difficulty` were swapped on one row
    (type="easy", difficulty="none"). Value-vocabulary checks catch it.
  * A sheet was copied from an older build, so its rows held different items
    than the answer key expected. Item-id and content checks catch it.
  * Excel silently strips a leading apostrophe and rewrites Bengali digits, so
    every text comparison here is apostrophe-, digit- and whitespace-insensitive
    to avoid raising false alarms about those.
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
from pathlib import Path

MAPPING = Path("data/annotated/round1/_mapping_DO_NOT_SHARE.csv")
CORPUS = Path("data/corpus/bn_v1/corpus.jsonl")

LABELS = {"correct", "wrong", "unsure", "unreadable"}
TYPES = {"none", "entity", "numeric", "relational", "contradiction",
         "fabricated", "overclaim"}
DIFFICULTY = {"easy", "hard"}

BN_DIGITS = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")


def norm(text: object) -> str:
    """Compare text the way a spreadsheet round-trip leaves it."""
    return (" ".join(str(text).split())
            .translate(BN_DIGITS)
            .lstrip("'")
            .rstrip("। . "))


def load_reference() -> tuple[dict, dict]:
    mapping = {}
    with MAPPING.open(encoding="utf-8-sig") as fh:
        for row in csv.DictReader(fh):
            mapping[row["item_id"]] = row
    corpus = {}
    with CORPUS.open(encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            corpus[r["id"]] = r
    return mapping, corpus


def check_file(path: Path, mapping: dict, corpus: dict) -> tuple[list[str], dict]:
    problems: list[str] = []
    stats = collections.Counter()

    with path.open(encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))

    if not rows:
        return [f"{path.name}: file is empty"], stats

    missing_cols = [c for c in ("item_id", "your_label", "your_type", "your_difficulty")
                    if c not in rows[0]]
    if missing_cols:
        return [f"{path.name}: missing column(s) {missing_cols}"], stats

    seen = collections.Counter(r["item_id"] for r in rows)
    for item_id, n in seen.items():
        if n > 1:
            problems.append(f"{path.name}: item_id {item_id} appears {n} times")

    for row in rows:
        item_id = (row.get("item_id") or "").strip()
        stats["rows"] += 1

        ref = mapping.get(item_id)
        if ref is None:
            problems.append(f"{path.name} [{item_id}]: unknown item_id "
                            "(sheet from a different build?)")
            continue

        # -- the row still describes the item it is supposed to describe -----
        rec = corpus.get(ref["record_id"])
        if rec is not None:
            if norm(row.get("question")) != norm(rec["question"]):
                problems.append(f"{path.name} [{item_id}]: question text was modified")
            if norm(row.get("answer")) != norm(rec["candidate_answer"]):
                problems.append(
                    f"{path.name} [{item_id}]: ANSWER text was modified - "
                    f"a label may have been typed into the answer cell "
                    f"(sheet={norm(row.get('answer'))[:40]!r})")

        label = (row.get("your_label") or "").strip().lower()
        vtype = (row.get("your_type") or "").strip().lower()
        diff = (row.get("your_difficulty") or "").strip().lower()

        if not label:
            stats["unlabelled"] += 1
            continue
        stats["labelled"] += 1
        stats[label] += 1

        if label not in LABELS:
            problems.append(f"{path.name} [{item_id}]: your_label={label!r} "
                            f"is not one of {sorted(LABELS)}")
        if vtype and vtype not in TYPES:
            hint = " (looks like a difficulty - are the two columns swapped?)" \
                if vtype in DIFFICULTY else ""
            problems.append(f"{path.name} [{item_id}]: your_type={vtype!r} invalid{hint}")
        if diff and diff not in DIFFICULTY:
            hint = " (looks like a type - are the two columns swapped?)" \
                if diff in TYPES else ""
            problems.append(f"{path.name} [{item_id}]: your_difficulty={diff!r} invalid{hint}")

        # -- label / type consistency ---------------------------------------
        if label == "correct" and vtype and vtype != "none":
            problems.append(f"{path.name} [{item_id}]: label=correct but type={vtype!r} "
                            "(a correct answer has type 'none')")
        if label == "wrong" and vtype == "none":
            problems.append(f"{path.name} [{item_id}]: label=wrong but type='none' "
                            "(a wrong answer needs a real type - D8)")
        if label in ("correct", "wrong"):
            # Incompleteness, not an error: a pre-filled sheet ships with the
            # ambiguous types deliberately blank for a human to fill. Counted
            # rather than listed one-per-row, so real errors stay visible.
            if not vtype:
                stats["needs_type"] += 1
            if not diff:
                stats["needs_difficulty"] += 1

        # -- type must match the condition ----------------------------------
        cond = ref["condition"]
        if label == "wrong" and vtype in TYPES and vtype != "none":
            intrinsic = {"entity", "numeric", "relational", "contradiction"}
            extrinsic = {"fabricated", "overclaim"}
            if cond == "has_context" and vtype in extrinsic:
                problems.append(f"{path.name} [{item_id}]: type={vtype!r} is a no-context "
                                "type but this item HAS a passage")
            if cond == "no_context" and vtype in intrinsic:
                problems.append(f"{path.name} [{item_id}]: type={vtype!r} is a has-context "
                                "type but this item has NO passage")

        if label in ("unsure", "unreadable") and not (row.get("your_notes") or "").strip():
            stats["unsure_without_note"] += 1

    return problems, stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, help="folder of sheets to check")
    ap.add_argument("--file", type=Path, nargs="*", default=[], help="individual sheets")
    args = ap.parse_args()

    paths: list[Path] = list(args.file)
    if args.dir:
        paths += sorted(p for p in args.dir.glob("*.csv") if not p.name.startswith("_"))
    if not paths:
        raise SystemExit("nothing to check - pass --dir or --file")

    if not MAPPING.exists():
        raise SystemExit(f"missing {MAPPING} - run src/build_annotation_sheets.py first")
    mapping, corpus = load_reference()

    all_problems: list[str] = []
    total = collections.Counter()
    print("=" * 66)
    for p in paths:
        problems, stats = check_file(p, mapping, corpus)
        total.update(stats)
        all_problems += problems
        done, rows = stats["labelled"], stats["rows"]
        pct = f"{done/rows:5.0%}" if rows else "   - "
        flag = "" if not problems else f"   <- {len(problems)} problem(s)"
        print(f"  {p.name:<28} {done:>4}/{rows:<4} labelled {pct}{flag}")

    print("=" * 66)
    print(f"  TOTAL {total['labelled']:,}/{total['rows']:,} rows labelled "
          f"({total['labelled']/max(total['rows'],1):.1%})")
    if total["rows"]:
        for k in ("correct", "wrong", "unsure", "unreadable"):
            if total[k]:
                print(f"    {k:<12}{total[k]:>6,}")
    if total["needs_type"] or total["needs_difficulty"]:
        print(f"  [ ] still to fill: {total['needs_type']:,} type(s), "
              f"{total['needs_difficulty']:,} difficulty(ies)")
    if total["unsure_without_note"]:
        print(f"  [i] {total['unsure_without_note']} unsure/unreadable row(s) have no note. "
              "A short reason helps (guidelines 2C).")

    if all_problems:
        print(f"\n*** {len(all_problems)} PROBLEM(S) ***")
        for x in all_problems[:40]:
            print("   ", x)
        if len(all_problems) > 40:
            print(f"    ... and {len(all_problems)-40} more")
        raise SystemExit(1)

    remaining = total["unlabelled"] + total["needs_type"] + total["needs_difficulty"]
    if remaining:
        print("")
        print(f"  No errors so far. {total['unlabelled']:,} row(s) still to label, "
              f"{total['needs_type']:,} type(s) still to fill.")
    else:
        print("")
        print("  No errors, and every row is complete. Ready to merge.")


if __name__ == "__main__":
    main()
