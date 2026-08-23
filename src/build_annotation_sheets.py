"""Build the M3 production annotation sheets (PRD D4 + D8).

Produces the sheets people actually fill in:

  test/  -- ALL 1,356 records, one folder per annotator. D4 requires the test
            split to be 100% human-verified and double-annotated, so both
            annotators label every row independently.
  dev/   -- ALL 1,346 records, single annotator.
  train_spotcheck/ -- a 20% sample of train (626 pairs / 1,252 records). Train
            is not labelled in full: models tolerate some label noise there, and
            the sample is enough to tell whether the split is sound.

Run from the repository root:

    python src/build_annotation_sheets.py            # build (refuses to clobber)
    python src/build_annotation_sheets.py --force    # rebuild, discarding work

PRE-FILL POLICY (why test is different)
---------------------------------------
dev and train_spotcheck arrive PRE-FILLED: the existing label, a derived
hallucination type where the rule is objective, and a derived difficulty. The
annotator verifies and corrects instead of typing from scratch.

The test split is NEVER pre-filled, not even its type column, and the reason is
worth stating plainly:

    `your_type` IS the binary label. It is "none" for every correct answer and
    a real type for every hallucinated one. Pre-filling it would tell the
    annotator the answer on all 1,356 rows and destroy the blind double
    annotation that PRD D4 requires.

So test costs what it costs. The saving comes from dev and train.

In the pre-filled sheets a random BLIND_SHARE of rows is left completely empty.
Those rows are the attention check: if corrections on the blind rows look like
corrections everywhere else, the annotator was reading rather than clicking
through. Which rows are blind is recorded in the private mapping.

Type derivation is deliberately conservative. It fills a type only where the
rule is objective -- the numbers differ, or a no-context answer is simply a
different answer -- and leaves the cell BLANK otherwise. Telling entity from
relational from contradiction needs a human, so it asks for one.

WHAT THIS FILE PROTECTS AGAINST
-------------------------------
Every guard below exists because the problem actually happened, or was found in
this corpus by measurement. None is hypothetical.

1. LABEL LEAKAGE THROUGH THE ROW ID. A corpus id looks like `bnp_007122_0`,
   and the suffix is the label: `_0` = correct, `_1` = hallucinated. Printing
   that in a sheet hands the annotator the answer key. Sheets therefore carry
   opaque ids (`test_0001`), and the id -> record mapping lives in a separate
   file the annotators never open.

2. PAIR MEMBERS SITTING NEXT TO EACH OTHER. The two answers to one question are
   a correct/hallucinated pair. Adjacent rows would let an annotator decide by
   comparing them instead of judging each on its own. Rows are shuffled and the
   separation is asserted before anything is written.

3. EXCEL EATING A LEADING APOSTROPHE. 1,056 questions and 15 answers start with
   a quote mark; Excel silently strips it on save ('জোয়ার' -> জোয়ার'). Nothing
   can stop that, so downstream comparison is apostrophe-insensitive. Recorded
   here so the next person does not treat it as corruption.

4. EXCEL EXECUTING A CELL AS A FORMULA. One answer in this corpus is `-1/4`.
   A cell beginning = + - @ is evaluated by Excel and can be rewritten or show
   an error. Those cells get a leading apostrophe, which Excel consumes as a
   text marker, leaving the value intact.

5. EMBEDDED NEWLINES AND TABS. 208 contexts and 2 questions contain them.
   Quoting alone survives Python's csv, but not every spreadsheet importer, so
   whitespace is collapsed. Downstream comparison is whitespace-insensitive, so
   this is safe.

6. OVERWRITING WORK IN PROGRESS. Rebuilding once destroyed a partly-filled
   sheet. The build now refuses to run if any target file already contains a
   label, unless --force is passed.

Deterministic: seed 42, byte-identical output on every rerun.
"""

from __future__ import annotations

import argparse
import collections
import csv
import difflib
import json
import random
import re
from pathlib import Path

CORPUS_SPLITS = Path("data/splits")
OUT_ROOT = Path("data/annotated/round1")

SEED = 42
BATCH_SIZE = 250          # rows per file: small enough to open, save and finish
TRAIN_SAMPLE = 0.20       # spot-check share of train
ANNOTATORS = ("tawhid", "shejan")

# test is excluded on purpose -- see PRE-FILL POLICY above.
PREFILL_SPLITS = {"dev", "train_spotcheck"}
BLIND_SHARE = 0.10        # rows left empty inside a pre-filled sheet

DIGITS = re.compile(r"[\d\u09e6-\u09ef]+")

SHEET_FIELDS = ["item_id", "condition", "context", "question", "answer",
                "your_label", "your_type", "your_difficulty", "your_notes"]

# A cell starting with one of these is a formula to Excel.
FORMULA_START = ("=", "+", "-", "@")


def clean(text: object) -> str:
    """Make a value safe to put in a spreadsheet cell without changing meaning."""
    s = " ".join(str(text).split())        # collapses newlines and tabs (guard 5)
    if s.startswith(FORMULA_START):        # guard 4
        s = "'" + s
    return s


def similarity(a: str, b: str) -> float:
    return difflib.SequenceMatcher(None, a, b).ratio()


def derive_type(correct_answer: str, wrong_answer: str, condition: str) -> str:
    """Hallucination type for the WRONG member of a pair, or "" to ask a human.

    Only objective rules fire. Everything requiring semantics is left blank:
    separating `entity` from `relational` from `contradiction` means reading the
    passage and understanding it, which is exactly the part a person is for.
    """
    ca = " ".join(str(correct_answer).split())
    wa = " ".join(str(wrong_answer).split())

    # The taxonomy is split by condition (PRD 5.2), and the split is not
    # cosmetic. `numeric` is an INTRINSIC type: it means the answer contradicts
    # a number in the passage. With no passage there is nothing to contradict,
    # so a wrong number in a no-context item is EXTRINSIC -- `fabricated`.
    # An earlier version ignored this and stamped `numeric` on no-context rows;
    # src/validate_annotation.py caught it.
    if condition == "has_context":
        nums_c, nums_w = DIGITS.findall(ca), DIGITS.findall(wa)
        if (nums_c or nums_w) and nums_c != nums_w:
            return "numeric"
        # entity / relational / contradiction need someone to read the passage.
        return ""

    # no-context: a different answer is a fabricated one. `overclaim` is the
    # rarer sibling and needs judgement, so near-identical answers stay blank.
    if similarity(ca, wa) < 0.90:
        return "fabricated"
    return ""


def derive_difficulty(correct_answer: str, wrong_answer: str) -> str:
    """easy / hard, using the definition the guidelines give the annotator:
    hard = the wrong answer is a close, believable near-miss."""
    ca = " ".join(str(correct_answer).split())
    wa = " ".join(str(wrong_answer).split())
    return "hard" if similarity(ca, wa) >= 0.60 else "easy"


def build_prefill(rows: list[dict], split_name: str,
                  rng: random.Random) -> tuple[dict[str, dict], set[str]]:
    """record_id -> {label, type, difficulty}, plus the set of blind record ids."""
    if split_name not in PREFILL_SPLITS:
        return {}, set()

    by_pair: dict[str, list[dict]] = collections.defaultdict(list)
    for r in rows:
        by_pair[r["pair_id"]].append(r)

    ids = sorted(r["id"] for r in rows)
    rng.shuffle(ids)
    blind = set(ids[:int(round(len(ids) * BLIND_SHARE))])

    prefill: dict[str, dict] = {}
    for r in rows:
        if r["id"] in blind:
            continue
        members = by_pair[r["pair_id"]]
        correct = next((m for m in members if m["label"] == 1), None)
        wrong = next((m for m in members if m["label"] == 0), None)
        if correct is None or wrong is None:
            continue
        ca, wa = correct["candidate_answer"], wrong["candidate_answer"]
        prefill[r["id"]] = {
            "label": "correct" if r["label"] == 1 else "wrong",
            "type": "none" if r["label"] == 1 else derive_type(ca, wa, r["condition"]),
            "difficulty": derive_difficulty(ca, wa),
        }
    return prefill, blind


def load_split(name: str) -> list[dict]:
    path = CORPUS_SPLITS / f"{name}.jsonl"
    return [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]


def order_rows(records: list[dict], rng: random.Random) -> list[dict]:
    """Order rows so the two members of a pair are never near each other, AND so
    a row's position says nothing about its label.

    An earlier version shuffled repeatedly and, when that failed to reach the
    target separation, fell back to "all first-members, then all second-members".
    Because the corpus writes the correct answer first, that fallback put every
    correct answer in the first half and every hallucinated answer in the second
    -- a worse leak than the problem it was solving. It shipped because the
    validation checked distance but never checked position-versus-label.

    The construction below cannot have that failure mode:

      * for each pair, a COIN FLIP decides which member goes to group A and
        which to group B, so neither group is correlated with the label;
      * each group is shuffled independently and A is placed before B, so the
        two members of a pair are separated by roughly len(A) rows.
    """
    by_pair: dict[str, list[dict]] = collections.defaultdict(list)
    for r in records:
        by_pair[r["pair_id"]].append(r)

    group_a: list[dict] = []
    group_b: list[dict] = []
    for pid in sorted(by_pair):                 # sorted => deterministic
        members = by_pair[pid][:]
        rng.shuffle(members)                    # the coin flip
        group_a.append(members[0])
        if len(members) > 1:
            group_b.extend(members[1:])

    rng.shuffle(group_a)
    rng.shuffle(group_b)
    return group_a + group_b


def write_batches(rows: list[dict], out_dir: Path, stem: str,
                  id_of: dict[str, str],
                  prefill: dict[str, dict] | None = None) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for n, start in enumerate(range(0, len(rows), BATCH_SIZE), start=1):
        chunk = rows[start:start + BATCH_SIZE]
        path = out_dir / f"{stem}_batch{n:02d}.csv"
        with path.open("w", encoding="utf-8-sig", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=SHEET_FIELDS)
            w.writeheader()
            for r in chunk:
                w.writerow({
                    "item_id": id_of[r["id"]],
                    "condition": r["condition"],
                    "context": "" if r["condition"] == "no_context" else clean(r["context"]),
                    "question": clean(r["question"]),
                    "answer": clean(r["candidate_answer"]),
                    "your_label": (prefill or {}).get(r["id"], {}).get("label", ""),
                    "your_type": (prefill or {}).get(r["id"], {}).get("type", ""),
                    "your_difficulty": (prefill or {}).get(r["id"], {}).get("difficulty", ""),
                    "your_notes": "",
                })
        written.append(path)
    return written


def existing_work(root: Path) -> list[Path]:
    """Any sheet under root that already has a label typed into it (guard 6)."""
    dirty: list[Path] = []
    for p in sorted(root.rglob("*.csv")):
        if p.name.startswith("_"):
            continue
        try:
            with p.open(encoding="utf-8-sig") as fh:
                for row in csv.DictReader(fh):
                    if (row.get("your_label") or "").strip():
                        dirty.append(p)
                        break
        except (OSError, csv.Error):
            continue
    return dirty


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true",
                    help="rebuild even though labelled sheets exist (DESTROYS them)")
    args = ap.parse_args()

    if OUT_ROOT.exists() and not args.force:
        dirty = existing_work(OUT_ROOT)
        if dirty:
            print(f"REFUSING TO BUILD: {len(dirty)} sheet(s) already contain labels.")
            for p in dirty[:10]:
                print(f"   {p}")
            print("\nRebuilding would destroy that work. Move it aside, or pass --force")
            print("if you really mean to discard it.")
            raise SystemExit(1)

    rng = random.Random(SEED)

    # ---- decide what goes into each sheet ---------------------------------
    test_rows = load_split("test")
    dev_rows = load_split("dev")

    train_all = load_split("train")
    by_pair: dict[str, list[dict]] = collections.defaultdict(list)
    for r in train_all:
        by_pair[r["pair_id"]].append(r)
    pair_ids = sorted(by_pair)                      # sorted first => deterministic
    rng.shuffle(pair_ids)
    n_sample = int(round(len(pair_ids) * TRAIN_SAMPLE))
    train_rows = [r for pid in pair_ids[:n_sample] for r in by_pair[pid]]

    jobs = [("test", test_rows), ("dev", dev_rows), ("train_spotcheck", train_rows)]

    prefills: dict[str, dict[str, dict]] = {}
    blinds: dict[str, set[str]] = {}
    for name, rows in jobs:
        prefills[name], blinds[name] = build_prefill(rows, name, rng)

    # ---- opaque ids, assigned after shuffling (guard 1) --------------------
    id_of: dict[str, str] = {}
    ordered: dict[str, list[dict]] = {}
    for name, rows in jobs:
        rows_ordered = order_rows(rows, rng)
        ordered[name] = rows_ordered
        prefix = {"test": "test", "dev": "dev", "train_spotcheck": "train"}[name]
        for i, r in enumerate(rows_ordered, start=1):
            id_of[r["id"]] = f"{prefix}_{i:04d}"

    # ---- validate before writing anything ---------------------------------
    problems: list[str] = []
    if len(set(id_of.values())) != len(id_of):
        problems.append("opaque item_ids are not unique")
    for name, rows in ordered.items():
        seen: dict[str, int] = {}
        worst = len(rows)
        for i, r in enumerate(rows):
            if r["pair_id"] in seen:
                worst = min(worst, i - seen[r["pair_id"]])
            seen[r["pair_id"]] = i
        if worst < 2:
            problems.append(f"{name}: pair members are adjacent (gap {worst})")

        # Position must not predict the label. This is the check whose absence
        # let the old first/second-member fallback ship a full label leak.
        pos_correct = [i for i, r in enumerate(rows) if r["label"] == 1]
        pos_wrong = [i for i, r in enumerate(rows) if r["label"] == 0]
        if pos_correct and pos_wrong:
            drift = abs(sum(pos_correct) / len(pos_correct)
                        - sum(pos_wrong) / len(pos_wrong))
            if drift > len(rows) * 0.08:
                problems.append(
                    f"{name}: row position predicts the label "
                    f"(mean position differs by {drift:.0f} of {len(rows)} rows)")
            half = len(rows) // 2
            first_half_correct = sum(1 for i in pos_correct if i < half)
            share = first_half_correct / max(len(pos_correct), 1)
            if not 0.40 <= share <= 0.60:
                problems.append(
                    f"{name}: {share:.0%} of correct answers land in the first half "
                    "- the halves are not label-balanced")
        for r in rows:
            if not clean(r["question"]) or not clean(r["candidate_answer"]):
                problems.append(f"{name}: empty question or answer on {r['id']}")
            if r["condition"] == "has_context" and not clean(r["context"]):
                problems.append(f"{name}: has_context record with empty context {r['id']}")
    # -- the pre-fill must never appear on test, and must be self-consistent --
    if prefills["test"] or blinds["test"]:
        problems.append("test split has pre-fill - it must stay blind (D4)")
    for name in ("dev", "train_spotcheck"):
        rows = ordered[name]
        pre = prefills[name]
        by_id = {r["id"]: r for r in rows}
        n_blind = len(blinds[name])
        share = n_blind / max(len(rows), 1)
        if not 0.07 <= share <= 0.13:
            problems.append(f"{name}: blind share {share:.1%} is outside 7-13%")
        for rid in blinds[name]:
            if rid in pre:
                problems.append(f"{name}: blind row {rid} was pre-filled anyway")
        for rid, vals in pre.items():
            rec = by_id[rid]
            if (vals["label"] == "correct") != (rec["label"] == 1):
                problems.append(f"{name}: pre-filled label disagrees with the corpus on {rid}")
            if (vals["type"] == "none") != (rec["label"] == 1):
                problems.append(f"{name}: type 'none' must mean correct - wrong on {rid}")
            if vals["type"] and vals["type"] not in (
                    "none", "entity", "numeric", "relational", "contradiction",
                    "fabricated", "overclaim"):
                problems.append(f"{name}: invalid pre-filled type {vals['type']!r} on {rid}")
            if rec["condition"] == "has_context" and vals["type"] in ("fabricated", "overclaim"):
                problems.append(f"{name}: no-context type on a has-context row {rid}")
            if rec["condition"] == "no_context" and vals["type"] in (
                    "entity", "relational", "contradiction"):
                problems.append(f"{name}: has-context type on a no-context row {rid}")
            if vals["difficulty"] not in ("easy", "hard"):
                problems.append(f"{name}: invalid pre-filled difficulty on {rid}")

    if problems:
        print("*** VALIDATION FAILED - nothing written ***")
        for p in problems[:10]:
            print("   ", p)
        raise SystemExit(1)

    # ---- write ------------------------------------------------------------
    written: list[Path] = []
    written += write_batches(ordered["test"], OUT_ROOT / "test" / ANNOTATORS[0],
                             "test", id_of, prefills["test"])
    written += write_batches(ordered["test"], OUT_ROOT / "test" / ANNOTATORS[1],
                             "test", id_of, prefills["test"])
    written += write_batches(ordered["dev"], OUT_ROOT / "dev", "dev", id_of,
                             prefills["dev"])
    written += write_batches(ordered["train_spotcheck"], OUT_ROOT / "train_spotcheck",
                             "train", id_of, prefills["train_spotcheck"])

    # ---- the private mapping (never given to annotators) -------------------
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    map_path = OUT_ROOT / "_mapping_DO_NOT_SHARE.csv"
    with map_path.open("w", encoding="utf-8-sig", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["item_id", "split", "record_id", "pair_id", "subject",
                    "condition", "difficulty", "existing_label",
                    "prefilled", "blind_check"])
        for name, rows in ordered.items():
            split = "test" if name == "test" else ("dev" if name == "dev" else "train")
            for r in rows:
                w.writerow([id_of[r["id"]], split, r["id"], r["pair_id"], r["subject"],
                            r["condition"], r["difficulty"], r["label"],
                            "yes" if r["id"] in prefills[name] else "no",
                            "yes" if r["id"] in blinds[name] else "no"])

    # ---- report -----------------------------------------------------------
    print(f"wrote {len(written)} sheet(s) under {OUT_ROOT}/\n")
    for name, rows in ordered.items():
        pairs = len({r["pair_id"] for r in rows})
        need_type = sum(1 for r in rows if r["label"] == 0)
        batches = (len(rows) + BATCH_SIZE - 1) // BATCH_SIZE
        who = "x2 annotators" if name == "test" else "x1 annotator"
        print(f"  {name:<16} {len(rows):>5,} rows  ({pairs:,} pairs)  {batches} batches  {who}")
        print(f"  {'':<16} of these {need_type:,} are hallucinated and need a type (D8)")
        if name in PREFILL_SPLITS:
            pre = prefills[name]
            typed = sum(1 for v in pre.values() if v["type"])
            print(f"  {'':<16} PRE-FILLED {len(pre):,} rows ({len(pre)/len(rows):.0%}); "
                  f"{len(rows)-len(pre):,} left empty "
                  f"({len(blinds[name]):,} blind checks + ambiguous types)")
            print(f"  {'':<16} of the pre-filled, {typed:,} carry a derived type")
        else:
            print(f"  {'':<16} NOT pre-filled - blind, because the type column "
                  "would reveal the label")
    total = len(ordered["test"]) * 2 + len(ordered["dev"]) + len(ordered["train_spotcheck"])
    print(f"\n  total human judgements: {total:,}")
    print(f"  private mapping: {map_path}  <- annotators must never open this")


if __name__ == "__main__":
    main()
