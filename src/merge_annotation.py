"""Write the finished human annotation into the corpus and the splits (PRD M3, D4, D8).

Reads the round-1 sheets people actually filled in:

    data/annotated/round1/test/tawhid/*_FINAL.csv     two annotators, blind (D4)
    data/annotated/round1/test/shejan/*_FINAL.csv
    data/annotated/round1/dev/*_FINAL.csv             one annotator
    data/annotated/round1/train_spotcheck/*_FINAL.csv one annotator, 20% of train
    data/annotated/round1/_mapping_DO_NOT_SHARE.csv   sheet item_id -> corpus record

and writes into `data/corpus/bn_v1/corpus.jsonl` and `data/splits/*.jsonl`:

    hallucination_type   the error type of a wrong answer (D8)
    type_source          where that type came from (see below)
    annotator_1/2        each human's binary label: 1 correct, 0 wrong, null unsure
    adjudicated          true when a disagreement was settled by the adjudicator
    excluded             true on BOTH records of a pair human review found unusable
    exclusion_reason     why, '+'-joined; empty when not excluded

Run from the repository root:

    python src/merge_annotation.py --dry-run       # report only, writes nothing
    python src/merge_annotation.py                 # write (refuses if adjudication is pending)
    python src/merge_annotation.py --allow-unreviewed
    python src/audit.py --data data/splits         # ALWAYS rerun after merging

WHAT IT WILL NEVER DO
---------------------
* Change `label`. A human who disagrees with the corpus label is recorded in
  `label_disputes.csv` and nothing else happens. The label is the prediction
  target; editing it from an annotation sheet would silently change what every
  model is scored against. A genuinely wrong label is fixed at the source and
  the corpus rebuilt.
* Change `difficulty`. That field is defined by a rule (does the string
  shortcut separate the pair?), not by opinion. Annotators' `your_difficulty`
  disagrees with it on ~40% of rows, which is expected: they are measuring a
  different thing.

HOW A TYPE IS DECIDED
---------------------
A correct answer (label 1) always gets `none`. That is the definition, not a
judgement.

A wrong answer (label 0) gets a type only when the humans settle it:

    test        both annotators said `wrong` AND chose the same type -> double_annotated
    dev, train  the annotator said `wrong` and chose a type          -> single_annotated

Every other wrong answer -- the two test annotators chose different types, one
or both said `correct`, someone marked `unsure` -- goes to
`data/annotated/round1/adjudication.csv`. Fill `your_decision` there with a type
(-> adjudicated), `skip` (leave unlabeled), or `dispute` (you believe the
corpus label itself is wrong; leave unlabeled and fix at the source). The file is
regenerated on every run and keeps the decisions already written in it.

SCOPE OF D8 (narrowed 2026-09-17, PRD §5.1c): a type is required for every wrong
answer in test, dev and the 20% train spot-check -- not for the other 80% of
train. Types are reporting metadata; no model trains on them, so typing train
records nobody reports on would cost ~2,500 annotations and change no result.
Those records stay `unlabeled` with `type_source = outside_train_sample`, which
marks them as out of scope by design rather than forgotten. (With `--with-llm`
and judge output from `src/llm_annotate.py` in `data/annotated/llm_round1/`,
they can be filled instead, marked `llm_consensus`. Not needed for D8.)

EXCLUDED PAIRS (PRD Q5 / §5.1d, decided 2026-09-17)
---------------------------------------------------
A pair is left out of BOTH training and scoring when human review showed its
stored labels cannot be trusted:

    wrong_answer_confirmed_correct  the adjudicator marked the stored-wrong answer `dispute`
    broken_item                     the adjudicator marked it `skip`
    correct_answer_judged_wrong     every annotator who saw the stored-correct answer
                                    said `wrong` (both on test, the one on dev/train)

The pair is dropped whole, never one record of it: removing one answer would
break the 50/50 balance and the pairing that splits rely on. Records stay in
the files, marked `excluded = true`; `src/splits.py` filters them out, so the
decision is applied the same way by every script and stays auditable. Only
human-checked pairs can be flagged, so the unchecked 80% of train keeps its
measured ~4.5% label noise.

Rerun this script after any `src/build_corpus.py` rebuild -- a rebuild resets
every type to `unlabeled`. It recomputes everything from the sheets each time,
so running it twice gives the same result.

NOTE: `hallucination_type`, `annotator_1/2` and `type_source` all reveal the
label. They are reporting metadata. Never feed them to a model.
"""

from __future__ import annotations

import argparse
import collections
import csv
import json
import shutil
from pathlib import Path

from validate_annotation import norm

ROUND1 = Path("data/annotated/round1")
MAPPING = ROUND1 / "_mapping_DO_NOT_SHARE.csv"
ADJUDICATION = ROUND1 / "adjudication.csv"
DISPUTES = ROUND1 / "label_disputes.csv"
LLM_DIR = Path("data/annotated/llm_round1")
CORPUS = Path("data/corpus/bn_v1/corpus.jsonl")
SPLITS = Path("data/splits")

INTRINSIC = {"entity", "numeric", "relational", "contradiction"}   # has_context
EXTRINSIC = {"fabricated", "overclaim"}                            # no_context
ALLOWED = {"has_context": INTRINSIC, "no_context": EXTRINSIC}
HUMAN_LABEL = {"correct": 1, "wrong": 0}   # unsure / unreadable / blank -> None


# -- reading ------------------------------------------------------------------

def read_sheets(folder: Path, split: str, mapping: dict, corpus: dict) -> dict[str, dict]:
    """record_id -> {item_id, label, type} for every row in a folder's *_FINAL.csv."""
    files = sorted(folder.glob("*_FINAL.csv"))
    if not files:
        raise SystemExit(f"{folder}: no *_FINAL.csv files")
    drafts = [p.name for p in folder.glob("*.csv")
              if not p.name.endswith("_FINAL.csv") and not p.name.startswith("_")]
    if drafts:
        raise SystemExit(f"{folder}: unfinished sheets next to the final ones: {drafts}. "
                         f"Finish or remove them so it is clear which version counts.")
    out: dict[str, dict] = {}
    for path in files:
        with path.open(encoding="utf-8-sig") as fh:
            for row in csv.DictReader(fh):
                iid = (row.get("item_id") or "").strip()
                if not iid:
                    continue
                m = mapping.get(iid)
                if m is None or m["split"] != split:
                    raise SystemExit(f"{path.name}: {iid} is not a {split} item in the mapping")
                rid = m["record_id"]
                if rid in out:
                    raise SystemExit(f"{folder}: {iid} appears twice")
                rec = corpus[rid]
                # a sheet from another build would describe a different question
                if (norm(row.get("question", "")) != norm(rec["question"])
                        or norm(row.get("answer", "")) != norm(rec["candidate_answer"])):
                    raise SystemExit(f"{path.name}: {iid} text does not match corpus record "
                                     f"{rid}. Run src/validate_annotation.py on {folder}.")
                out[rid] = {
                    "item_id": iid,
                    "label": HUMAN_LABEL.get((row.get("your_label") or "").strip().lower()),
                    "raw_label": (row.get("your_label") or "").strip().lower() or "blank",
                    "type": (row.get("your_type") or "").strip().lower(),
                }
    return out


def read_decisions() -> dict[str, tuple[str, str]]:
    """item_id -> (your_decision, adjudicator_notes) already written by a human."""
    if not ADJUDICATION.exists():
        return {}
    with ADJUDICATION.open(encoding="utf-8-sig") as fh:
        return {r["item_id"]: ((r.get("your_decision") or "").strip().lower(),
                               r.get("adjudicator_notes") or "")
                for r in csv.DictReader(fh)}


def read_llm_types() -> dict[str, str]:
    """record_id -> type accepted by judge consensus in src/llm_annotate.py."""
    types: dict[str, str] = {}
    path = LLM_DIR / "proposals.jsonl"
    if not path.exists():
        raise SystemExit(f"--with-llm given but {path} does not exist")
    for line in path.open(encoding="utf-8"):
        if line.strip():
            p = json.loads(line)
            if p.get("task") == "type" and p.get("accepted") and p.get("proposed"):
                types[p["id"]] = p["proposed"]
    return types


# -- deciding -----------------------------------------------------------------

def cell(text: object) -> str:
    """Spreadsheet-safe text, same guards as build_annotation_sheets.py: no
    embedded newlines, and no leading = + - @ for Excel to run as a formula."""
    s = " ".join(str(text).split())
    return "'" + s if s.startswith(("=", "+", "-", "@")) else s


def describe(votes: list[tuple[str, dict]]) -> str:
    return "; ".join(f"{who}={v['raw_label']}/{v['type'] or '-'}" for who, v in votes)


def settle(rec: dict, votes: list[tuple[str, dict]]) -> tuple[str | None, str]:
    """For a wrong answer: (type, source) if the humans settled it, else (None, reason)."""
    labels = [v["label"] for _, v in votes]
    types = {v["type"] for _, v in votes}
    if all(l == 0 for l in labels) and len(types) == 1:
        t = types.pop()
        if t in ALLOWED[rec["condition"]]:
            return t, "double_annotated" if len(votes) == 2 else "single_annotated"
        return None, f"type {t!r} is not valid for {rec['condition']}"
    if all(l == 0 for l in labels):
        return None, "annotators chose different types"
    if any(l is None for l in labels):
        return None, "marked unsure or unreadable"
    if all(l == 1 for l in labels):
        return None, "human says correct, corpus says wrong"
    return None, "annotators disagree on correct vs wrong"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true", help="report only; write nothing")
    ap.add_argument("--allow-unreviewed", action="store_true",
                    help="write even though adjudication rows are undecided "
                         "(those records stay 'unlabeled')")
    ap.add_argument("--with-llm", action="store_true",
                    help="fill wrong answers no human sheet covers from llm_round1 judge consensus")
    ap.add_argument("--test-annotators", default="tawhid,shejan",
                    help="folder names under test/, in annotator_1,annotator_2 order")
    args = ap.parse_args()

    corpus_rows = [json.loads(l) for l in CORPUS.open(encoding="utf-8") if l.strip()]
    corpus = {r["id"]: r for r in corpus_rows}
    with MAPPING.open(encoding="utf-8-sig") as fh:
        mapping = {r["item_id"]: r for r in csv.DictReader(fh)}

    # the mapping must still describe these splits; a rebuilt corpus with a
    # different split would make every sheet point at the wrong record
    split_ids = {s: {json.loads(l)["id"] for l in (SPLITS / f"{s}.jsonl").open(encoding="utf-8")}
                 for s in ("train", "dev", "test")}
    stale = [m["item_id"] for m in mapping.values() if m["record_id"] not in split_ids[m["split"]]]
    if stale:
        raise SystemExit(f"{len(stale)} mapping rows point at records no longer in their split "
                         f"(e.g. {stale[:3]}). The corpus was rebuilt differently; stop.")

    # -- collect every human vote, per record ---------------------------------
    first, second = [a.strip() for a in args.test_annotators.split(",")]
    sources = [(first, ROUND1 / "test" / first, "test"),
               (second, ROUND1 / "test" / second, "test"),
               ("annotator", ROUND1 / "dev", "dev"),
               ("annotator", ROUND1 / "train_spotcheck", "train")]
    votes: dict[str, list[tuple[str, dict]]] = collections.defaultdict(list)
    for who, folder, split in sources:
        for rid, v in read_sheets(folder, split, mapping, corpus).items():
            votes[rid].append((who, v))
    split_of = {m["record_id"]: m["split"] for m in mapping.values()}
    for rid, vs in votes.items():
        need = 2 if split_of[rid] == "test" else 1
        if len(vs) != need:
            raise SystemExit(f"{rid}: expected {need} annotator(s), found {len(vs)}")

    decisions = read_decisions()
    llm_types = read_llm_types() if args.with_llm else {}

    # -- decide every record --------------------------------------------------
    new: dict[str, dict] = {}
    pending, dispute_rows, bad_decisions = [], [], []
    for r in corpus_rows:
        rid, vs = r["id"], votes.get(r["id"], [])
        fields = {"annotator_1": vs[0][1]["label"] if vs else None,
                  "annotator_2": vs[1][1]["label"] if len(vs) > 1 else None,
                  "adjudicated": False}

        if vs and any(v["label"] is not None and v["label"] != r["label"] for _, v in vs):
            dispute_rows.append((r, vs))

        if r["label"] == 1:
            new[rid] = {**fields, "hallucination_type": "none", "type_source": "definitional"}
            continue

        if not vs:
            t = llm_types.get(rid)
            if t and t in ALLOWED[r["condition"]]:
                new[rid] = {**fields, "hallucination_type": t, "type_source": "llm_consensus"}
            else:
                new[rid] = {**fields, "hallucination_type": "unlabeled",
                            "type_source": "outside_train_sample"}
            continue

        t, why = settle(r, vs)
        if t:
            new[rid] = {**fields, "hallucination_type": t, "type_source": why}
            continue

        iid = vs[0][1]["item_id"]
        decision, _ = decisions.get(iid, ("", ""))
        pending.append((r, vs, why))
        if decision in ALLOWED[r["condition"]]:
            new[rid] = {**fields, "adjudicated": True,
                        "hallucination_type": decision, "type_source": "adjudicated"}
        elif decision in ("skip", "dispute"):
            new[rid] = {**fields, "hallucination_type": "unlabeled", "type_source": f"adjudicator_{decision}"}
        else:
            if decision:
                bad_decisions.append(f"{iid}: {decision!r} is not one of "
                                     f"{sorted(ALLOWED[r['condition']])} or skip/dispute")
            new[rid] = {**fields, "hallucination_type": "unlabeled", "type_source": "awaiting_adjudication"}

    # -- PRD Q5: pairs whose labels human review could not trust ----------------
    pair_reasons: dict[str, set[str]] = collections.defaultdict(set)
    for r in corpus_rows:
        src, vs = new[r["id"]]["type_source"], votes.get(r["id"], [])
        if src == "adjudicator_dispute":
            pair_reasons[r["pair_id"]].add("wrong_answer_confirmed_correct")
        elif src == "adjudicator_skip":
            pair_reasons[r["pair_id"]].add("broken_item")
        if r["label"] == 1 and vs and all(v["label"] == 0 for _, v in vs):
            pair_reasons[r["pair_id"]].add("correct_answer_judged_wrong")
    for r in corpus_rows:
        reasons = sorted(pair_reasons.get(r["pair_id"], ()))
        new[r["id"]]["excluded"] = bool(reasons)
        new[r["id"]]["exclusion_reason"] = "+".join(reasons)

    if bad_decisions:
        print(f"*** {len(bad_decisions)} invalid decision(s) in {ADJUDICATION}; nothing written ***")
        for b in bad_decisions[:10]:
            print("   ", b)
        raise SystemExit(1)

    # -- write the two human-facing sheets (always; they hold no corpus data) --
    with ADJUDICATION.open("w", encoding="utf-8-sig", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["item_id", "split", "condition", "subject", "reason", "corpus_label",
                    "votes", "allowed_types", "context", "question", "answer",
                    "your_decision", "adjudicator_notes"])
        for r, vs, why in sorted(pending, key=lambda x: x[1][0][1]["item_id"]):
            iid = vs[0][1]["item_id"]
            d, notes = decisions.get(iid, ("", ""))
            w.writerow([iid, split_of[r["id"]], r["condition"], r["subject"], why, "wrong",
                        describe(vs), " | ".join(sorted(ALLOWED[r["condition"]])),
                        cell(r["context"]), cell(r["question"]), cell(r["candidate_answer"]),
                        d, notes])
    with DISPUTES.open("w", encoding="utf-8-sig", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["item_id", "split", "condition", "subject", "corpus_label", "votes",
                    "all_humans_disagree_with_corpus", "context", "question", "answer"])
        for r, vs in sorted(dispute_rows, key=lambda x: x[1][0][1]["item_id"]):
            unanimous = all(v["label"] is not None and v["label"] != r["label"] for _, v in vs)
            w.writerow([vs[0][1]["item_id"], split_of[r["id"]], r["condition"], r["subject"],
                        "correct" if r["label"] else "wrong", describe(vs),
                        "yes" if unanimous else "no",
                        cell(r["context"]), cell(r["question"]), cell(r["candidate_answer"])])

    # -- report ---------------------------------------------------------------
    undecided = sum(1 for v in new.values() if v["type_source"] == "awaiting_adjudication")
    print("=" * 70)
    print("MERGE REPORT - human annotation -> corpus")
    print("=" * 70)
    print(f"  {'split':<7}{'wrong answers':>14}{'typed':>8}{'coverage':>10}   type sources")
    for split in ("train", "dev", "test"):
        wrong = [rid for rid in split_ids[split] if corpus[rid]["label"] == 0]
        typed = [rid for rid in wrong if new[rid]["hallucination_type"] != "unlabeled"]
        src = collections.Counter(new[rid]["type_source"] for rid in wrong)
        print(f"  {split:<7}{len(wrong):>14,}{len(typed):>8,}{len(typed)/len(wrong):>10.1%}   "
              + ", ".join(f"{k} {v:,}" for k, v in src.most_common()))
    wrong_all = [r for r in corpus_rows if r["label"] == 0]
    typed_all = sum(1 for r in wrong_all if new[r["id"]]["hallucination_type"] != "unlabeled")
    print(f"  {'all':<7}{len(wrong_all):>14,}{typed_all:>8,}{typed_all/len(wrong_all):>10.1%}")

    # D8 covers test + dev + the train spot-check (PRD §5.1c), i.e. every wrong
    # answer some human looked at. Adjudicator skip/dispute are reported
    # exclusions, not gaps: those items cannot be honestly typed.
    in_scope = [r for r in wrong_all if new[r["id"]]["type_source"] != "outside_train_sample"]
    scope_src = collections.Counter(new[r["id"]]["type_source"] for r in in_scope)
    in_scope_typed = sum(1 for r in in_scope if new[r["id"]]["hallucination_type"] != "unlabeled")
    excluded = scope_src["adjudicator_skip"] + scope_src["adjudicator_dispute"]
    waiting = scope_src["awaiting_adjudication"]
    print(f"\n  D8 scope (test + dev + train sample): {len(in_scope):,} wrong answers, "
          f"{in_scope_typed:,} typed, {excluded:,} excluded by adjudicator, "
          f"{waiting:,} awaiting adjudication -> D8 {'MET' if waiting == 0 else 'NOT MET'}")

    print("\n  type distribution among typed wrong answers:")
    for cond in ("has_context", "no_context"):
        c = collections.Counter(new[r["id"]]["hallucination_type"] for r in wrong_all
                                if r["condition"] == cond and new[r["id"]]["hallucination_type"] != "unlabeled")
        print(f"    {cond:<12} " + ", ".join(f"{k} {v:,}" for k, v in c.most_common()))

    reasons = collections.Counter((split_of[r["id"]], why) for r, _, why in pending)
    print(f"\n  needs adjudication: {len(pending):,} ({undecided:,} still undecided) -> {ADJUDICATION}")
    for (split, why), n in sorted(reasons.items()):
        print(f"    {split:<6}{why:<45}{n:>5,}")
    unanimous = sum(1 for r, vs in dispute_rows
                    if all(v["label"] is not None and v["label"] != r["label"] for _, v in vs))
    print(f"\n  label disputes: {len(dispute_rows):,} records where a human disagrees with the "
          f"corpus label ({unanimous:,} where every annotator does) -> {DISPUTES}")
    print("    Labels are never changed here. These are a corpus finding to report.")

    print("\n  excluded from training and scoring (PRD Q5):")
    for split in ("train", "dev", "test"):
        pids = {corpus[rid]["pair_id"] for rid in split_ids[split]}
        out = {p for p in pids if p in pair_reasons}
        why = collections.Counter(r for p in out for r in pair_reasons[p])
        print(f"    {split:<6}{len(out):>4} of {len(pids):>5,} pairs -> {len(pids)-len(out):>5,} usable   "
              + ", ".join(f"{k} {v}" for k, v in why.most_common()))

    if args.dry_run:
        print("\n  DRY RUN - corpus and splits untouched.")
        print("=" * 70)
        return
    if undecided and not args.allow_unreviewed:
        print("=" * 70)
        raise SystemExit(f"{undecided:,} adjudication row(s) have no decision. Fill `your_decision` "
                         f"in {ADJUDICATION}, or rerun with --allow-unreviewed to leave them unlabeled.")

    # -- write ----------------------------------------------------------------
    for path in [CORPUS] + [SPLITS / f"{s}.jsonl" for s in ("train", "dev", "test")]:
        backup = path.with_suffix(path.suffix + ".bak")
        shutil.copy(path, backup)
        recs = [json.loads(l) for l in path.open(encoding="utf-8") if l.strip()]
        before = [(r["id"], r["label"], r["difficulty"]) for r in recs]
        for r in recs:
            r.update(new[r["id"]])
        assert before == [(r["id"], r["label"], r["difficulty"]) for r in recs], \
            "label or difficulty changed - refusing to write"
        with path.open("w", encoding="utf-8", newline="\n") as fh:
            for r in recs:
                fh.write(json.dumps(r, ensure_ascii=False, sort_keys=True) + "\n")
        print(f"  wrote {path}   (backup: {backup.name})")
    print("\n  Now rerun the gate:  python src/audit.py --data data/splits")
    print("=" * 70)


if __name__ == "__main__":
    main()
