"""Score the 100-item agreement test (PRD M2 gate: Cohen's kappa >= 0.60).

Compares two annotators' filled-in sheets against each other, and each of them
against the answer key. Run it only AFTER both annotators have submitted.

    python src/score_agreement.py --a data/annotated/agreement_test_v1/items_tawhidul.csv \\
                                  --b data/annotated/agreement_test_v1/items_friend.csv

Cohen's kappa measures agreement ABOVE what two people would hit by chance.
Plain percent-agreement is not enough: on a 50/50 task two people guessing at
random already agree half the time, so 75% agreement is much weaker than it
sounds. Kappa subtracts that chance level -- 0 means no better than guessing,
1 means perfect.
"""

from __future__ import annotations

import argparse
import collections
import csv
from pathlib import Path

GATE = 0.60
KEY = Path("data/annotated/agreement_test_v1/answer_key_DO_NOT_OPEN_YET.csv")


def read_labels(path: Path) -> dict[str, str]:
    with path.open(encoding="utf-8-sig") as fh:
        rows = list(csv.DictReader(fh))
    out = {}
    for r in rows:
        v = (r.get("your_label") or "").strip().lower()
        if v in ("1", "correct", "ok", "right"):
            out[r["item_id"]] = "correct"
        elif v in ("0", "wrong", "hallucinated", "incorrect"):
            out[r["item_id"]] = "wrong"
        elif v in ("unsure", "unknown", "?"):
            out[r["item_id"]] = "unsure"
        elif v:
            out[r["item_id"]] = v          # e.g. "unreadable"
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
    ap.add_argument("--a", required=True, type=Path, help="first annotator's sheet")
    ap.add_argument("--b", required=True, type=Path, help="second annotator's sheet")
    args = ap.parse_args()

    A, B = read_labels(args.a), read_labels(args.b)
    with KEY.open(encoding="utf-8-sig") as fh:
        key = {r["item_id"]: r for r in csv.DictReader(fh)}

    # Content-integrity check. A sheet copied from an older build can carry a
    # DIFFERENT item under the same item_id. Scoring that against this key would
    # silently compare labels for two different questions, so any row whose
    # question+answer does not match the corpus record the key points at is
    # dropped and reported.
    stale: list[str] = []
    corpus_path = Path("data/corpus/bn_v1/corpus.jsonl")
    if corpus_path.exists():
        import json
        corpus: dict[str, list] = {}
        for line in corpus_path.open(encoding="utf-8"):
            r = json.loads(line)
            corpus.setdefault(r["pair_id"], []).append(r)
        # Bengali and ASCII digits are the same value written two ways, and an
        # annotator retyping a cell may use either. Fold them together before
        # comparing, or "৩" and "3" look like different items.
        BN_DIGITS = str.maketrans("০১২৩৪৫৬৭৮৯", "0123456789")

        def norm(t: str) -> str:
            return (" ".join(str(t).split())
                    .translate(BN_DIGITS)
                    .lstrip("'")
                    .rstrip("।. "))
        sheets = {}
        for path in (args.a, args.b):
            with path.open(encoding="utf-8-sig") as fh:
                sheets[path] = {r["item_id"]: r for r in csv.DictReader(fh)}
        for item_id, k in key.items():
            grp = corpus.get(k["source_pair_id"], [])
            want = [m for m in grp if str(m["label"]) == k["true_label"]]
            if not want:
                stale.append(item_id)
                continue
            for path, rows in sheets.items():
                row = rows.get(item_id)
                if row and (norm(row["question"]) != norm(want[0]["question"])
                            or norm(row["answer"]) != norm(want[0]["candidate_answer"])):
                    if item_id not in stale:
                        stale.append(item_id)
        if stale:
            print(f"[!] {len(stale)} item(s) EXCLUDED - the sheet content does not match this")
            print(f"    answer key (a sheet from an older build): {sorted(stale)}")
            print("    Their labels describe a different question and cannot be scored.")
            print("")

    both = sorted(i for i in (set(A) & set(B) & set(key)) if i not in set(stale))
    missing_a = sorted(set(key) - set(A))
    missing_b = sorted(set(key) - set(B))
    if missing_a:
        print(f"[!] {args.a.name} is missing {len(missing_a)} items: {missing_a[:5]}...")
    if missing_b:
        print(f"[!] {args.b.name} is missing {len(missing_b)} items: {missing_b[:5]}...")
    if not both:
        raise SystemExit("no overlapping labelled items")

    # Items either annotator marked unsure/unreadable are excluded from kappa and
    # reported separately. Scoring a guess against a non-answer would understate
    # agreement and hide which subjects are actually hard to verify.
    SKIP = {"unsure", "unreadable"}
    skipped = [i for i in both if A[i] in SKIP or B[i] in SKIP]
    both = [i for i in both if i not in set(skipped)]
    if skipped:
        bysub = collections.Counter(key[i]["subject"] for i in skipped)
        print(f"[i] {len(skipped)} item(s) excluded as UNLABELABLE (unsure/unreadable): {dict(bysub)}")
        print("    These are items an annotator could not label honestly -- the passage")
        print("    does not answer the question, the answer type does not match the")
        print("    question, or the text is broken. A cluster in one subject is a")
        print("    defect in the corpus, not a failure by the annotators.")
        print("")
    if not both:
        raise SystemExit("every item was marked unsure or unreadable")

    a = [A[i] for i in both]
    b = [B[i] for i in both]
    truth = [key[i]["true_label_word"] for i in both]

    k = cohens_kappa(a, b)
    agree = sum(1 for x, y in zip(a, b) if x == y)
    print("=" * 62)
    print(f"AGREEMENT TEST  ({len(both)} items labelled by both)")
    print("=" * 62)
    print(f"  raw agreement        : {agree}/{len(both)} = {agree/len(both):.1%}")
    print(f"  Cohen's kappa        : {k:.3f}   (gate: >= {GATE})")
    print(f"  annotator A vs key   : {sum(1 for x,t in zip(a,truth) if x==t)/len(both):.1%} match")
    print(f"  annotator B vs key   : {sum(1 for x,t in zip(b,truth) if x==t)/len(both):.1%} match")

    print("\n  agreement by slice:")
    for field in ("condition", "difficulty"):
        for val in sorted({key[i][field] for i in both}):
            idx = [n for n, i in enumerate(both) if key[i][field] == val]
            hit = sum(1 for n in idx if a[n] == b[n])
            print(f"    {field:<11}{val:<14}{hit}/{len(idx)} = {hit/len(idx):.0%}")

    dis = [i for n, i in enumerate(both) if a[n] != b[n]]
    if dis:
        print(f"\n  {len(dis)} DISAGREEMENTS -- these are what to fix in the guidelines:")
        for i in dis[:20]:
            print(f"    {i}  {key[i]['condition']:<12} {key[i]['difficulty']:<5} "
                  f"truth={key[i]['true_label_word']:<8} A={A[i]:<9} B={B[i]}")

    print("\n" + "=" * 62)
    if k >= GATE:
        print(f"GATE M2 PASSED: kappa {k:.3f} >= {GATE}")
    else:
        print(f"GATE M2 FAILED: kappa {k:.3f} < {GATE}")
        print("Fix the GUIDELINES, not the annotators. Write an explicit rule for")
        print("each disagreement above, then rerun on the same 100 items.")
    print("=" * 62)
    raise SystemExit(0 if k >= GATE else 1)


if __name__ == "__main__":
    main()
