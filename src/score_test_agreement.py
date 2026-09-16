"""Score the full test-split agreement between two annotators (PRD D4/D5).

`src/score_agreement.py` only handles the 100-item M2 pilot: a single file per
annotator, and a fixed answer key with only 100 rows. This script is for the
real thing -- the full 1,356-item test split, spread across 6 batch files per
annotator, with no separate answer key. The "key" here is the private mapping
`data/annotated/round1/_mapping_DO_NOT_SHARE.csv`, which carries each item's
condition/difficulty/subject and the corpus's existing binary label.

    python src/score_test_agreement.py \\
        --a-dir data/annotated/round1/test/tawhid \\
        --b-dir data/annotated/round1/test/shejan

Cohen's kappa measures agreement ABOVE what two people would hit by chance.
Plain percent-agreement overstates things -- on a near-balanced task two people
guessing at random already agree close to half the time.
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
from pathlib import Path

GATE = 0.60
MAPPING = Path("data/annotated/round1/_mapping_DO_NOT_SHARE.csv")
CORPUS = Path("data/corpus/bn_v1/corpus.jsonl")

BN_DIGITS = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")


def norm(t: object) -> str:
    return (" ".join(str(t).split()).translate(BN_DIGITS)
            .lstrip("'").rstrip("।. "))


def read_labels(folder: Path) -> dict[str, dict]:
    """item_id -> {label, type, difficulty, question, answer} across all
    batch files in the folder, whatever their names."""
    out: dict[str, dict] = {}
    files = sorted(p for p in folder.glob("*.csv") if not p.name.startswith("_"))
    if not files:
        raise SystemExit(f"no CSV files found in {folder}")
    for path in files:
        with path.open(encoding="utf-8-sig") as fh:
            for row in csv.DictReader(fh):
                iid = row.get("item_id", "").strip()
                if not iid:
                    continue
                if iid in out:
                    raise SystemExit(f"{path}: item_id {iid} also appears in "
                                     f"another file in {folder} -- duplicate row")
                v = (row.get("your_label") or "").strip().lower()
                if v in ("1", "correct", "ok", "right"):
                    label = "correct"
                elif v in ("0", "wrong", "hallucinated", "incorrect"):
                    label = "wrong"
                elif v in ("unsure", "unknown", "?"):
                    label = "unsure"
                elif v:
                    label = v
                else:
                    label = None
                out[iid] = {
                    "label": label,
                    "type": (row.get("your_type") or "").strip().lower(),
                    "difficulty": (row.get("your_difficulty") or "").strip().lower(),
                    "question": row.get("question", ""),
                    "answer": row.get("answer", ""),
                    "source_file": path.name,
                }
    return out


def cohens_kappa(a: list[str], b: list[str]) -> float:
    labels = sorted(set(a) | set(b))
    n = len(a)
    observed = sum(1 for x, y in zip(a, b) if x == y) / n
    ca, cb = collections.Counter(a), collections.Counter(b)
    expected = sum((ca[l] / n) * (cb[l] / n) for l in labels)
    return 1.0 if expected == 1 else (observed - expected) / (1 - expected)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a-dir", required=True, type=Path)
    ap.add_argument("--b-dir", required=True, type=Path)
    ap.add_argument("--a-name", default="A")
    ap.add_argument("--b-name", default="B")
    args = ap.parse_args()

    if not MAPPING.exists():
        raise SystemExit(f"missing {MAPPING} -- run src/build_annotation_sheets.py first")
    with MAPPING.open(encoding="utf-8-sig") as fh:
        mapping = {r["item_id"]: r for r in csv.DictReader(fh)}
    # only score the test split, in case this mapping ever covers more
    mapping = {k: v for k, v in mapping.items() if v["split"] == "test"}

    corpus = {}
    if CORPUS.exists():
        for line in CORPUS.open(encoding="utf-8"):
            r = json.loads(line)
            corpus[r["id"]] = r

    A = read_labels(args.a_dir)
    B = read_labels(args.b_dir)

    print(f"{args.a_name}: {len(A):,} rows across "
          f"{len({v['source_file'] for v in A.values()})} file(s)")
    print(f"{args.b_name}: {len(B):,} rows across "
          f"{len({v['source_file'] for v in B.values()})} file(s)")

    missing_a = sorted(set(mapping) - set(A))
    missing_b = sorted(set(mapping) - set(B))
    extra_a = sorted(set(A) - set(mapping))
    extra_b = sorted(set(B) - set(mapping))
    if missing_a:
        print(f"[!] {args.a_name} is missing {len(missing_a)} test item(s): {missing_a[:5]}...")
    if missing_b:
        print(f"[!] {args.b_name} is missing {len(missing_b)} test item(s): {missing_b[:5]}...")
    if extra_a:
        print(f"[!] {args.a_name} has {len(extra_a)} item(s) not in the test split "
              f"(different batch?): {extra_a[:5]}...")
    if extra_b:
        print(f"[!] {args.b_name} has {len(extra_b)} item(s) not in the test split: {extra_b[:5]}...")

    # -- content-integrity: does each row still describe the item the mapping
    # says it should? Catches a sheet copied from an older/wrong build. --------
    stale: set[str] = set()
    for iid, ref in mapping.items():
        rec = corpus.get(ref["record_id"])
        if rec is None:
            continue
        for name, sheet in ((args.a_name, A), (args.b_name, B)):
            row = sheet.get(iid)
            if row and (norm(row["question"]) != norm(rec["question"])
                        or norm(row["answer"]) != norm(rec["candidate_answer"])):
                stale.add(iid)
    if stale:
        print(f"\n[!] {len(stale)} item(s) EXCLUDED -- sheet content does not match "
              f"the corpus record the mapping points at:")
        print(f"    {sorted(stale)[:10]}")
        print("    These describe a different question in at least one sheet and "
              "cannot be scored.\n")

    both = sorted(i for i in (set(A) & set(B) & set(mapping)) if i not in stale)
    if not both:
        raise SystemExit("no overlapping scorable items")

    # -- drop unsure/unreadable from either side -----------------------------
    SKIP = {"unsure", "unreadable", None}
    skipped = [i for i in both if A[i]["label"] in SKIP or B[i]["label"] in SKIP]
    scored = [i for i in both if i not in set(skipped)]
    if skipped:
        bysub = collections.Counter(mapping[i]["subject"] for i in skipped)
        print(f"[i] {len(skipped)} item(s) excluded as unsure/unreadable/unlabelled "
              f"(by either annotator): {dict(bysub.most_common())}")
        print("    A cluster in one subject is a corpus finding, not an annotator failure.\n")
    if not scored:
        raise SystemExit("every overlapping item was unsure/unreadable/unlabelled")

    a = [A[i]["label"] for i in scored]
    b = [B[i]["label"] for i in scored]
    corpus_label = [("correct" if mapping[i]["existing_label"] == "1" else "wrong")
                    for i in scored]

    k = cohens_kappa(a, b)
    agree = sum(1 for x, y in zip(a, b) if x == y)

    print("=" * 66)
    print(f"TEST-SPLIT AGREEMENT  ({len(scored):,} items scored by both, "
          f"of {len(mapping):,} in the test split)")
    print("=" * 66)
    print(f"  raw agreement                 : {agree:,}/{len(scored):,} = {agree/len(scored):.1%}")
    print(f"  Cohen's kappa                 : {k:.3f}   (reference gate: >= {GATE})")
    print(f"  {args.a_name} vs existing corpus label : "
          f"{sum(1 for x,c in zip(a,corpus_label) if x==c)/len(scored):.1%} match")
    print(f"  {args.b_name} vs existing corpus label : "
          f"{sum(1 for x,c in zip(b,corpus_label) if x==c)/len(scored):.1%} match")

    print("\n  agreement by slice:")
    for field in ("condition", "difficulty", "subject"):
        print(f"  [{field}]")
        for val in sorted({mapping[i][field] for i in scored}):
            idx = [n for n, i in enumerate(scored) if mapping[i][field] == val]
            hit = sum(1 for n in idx if a[n] == b[n])
            print(f"    {val:<24}{hit:>5,}/{len(idx):<5,} = {hit/len(idx):.0%}")

    dis = [i for n, i in enumerate(scored) if a[n] != b[n]]
    print(f"\n  {len(dis):,} disagreements out of {len(scored):,} scored items "
          f"({len(dis)/len(scored):.1%})")
    if dis:
        print("  first 20, for spot-checking:")
        for i in dis[:20]:
            m = mapping[i]
            print(f"    {i}  {m['condition']:<12}{m['subject']:<20}"
                  f"corpus={('correct' if m['existing_label']=='1' else 'wrong'):<8}"
                  f"{args.a_name}={A[i]['label']:<9}{args.b_name}={B[i]['label']}")

    out_path = Path("data/annotated/round1/test_disagreements.csv")
    with out_path.open("w", encoding="utf-8-sig", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["item_id", "condition", "subject", "difficulty",
                    "corpus_label", f"{args.a_name}_label", f"{args.b_name}_label",
                    "question", "answer"])
        for i in dis:
            m = mapping[i]
            rec = corpus.get(m["record_id"], {})
            w.writerow([i, m["condition"], m["subject"], m["difficulty"],
                       "correct" if m["existing_label"] == "1" else "wrong",
                       A[i]["label"], B[i]["label"],
                       rec.get("question", ""), rec.get("candidate_answer", "")])
    print(f"\n  full disagreement list written to {out_path}")

    print("\n" + "=" * 66)
    if k >= GATE:
        band = "almost perfect" if k > 0.80 else "substantial"
        print(f"kappa {k:.3f} >= {GATE}  ({band} agreement on the Landis-Koch scale)")
    else:
        print(f"kappa {k:.3f} < {GATE}")
    print("=" * 66)


if __name__ == "__main__":
    main()
