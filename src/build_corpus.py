"""Build the Phase 1 Bengali corpus: interim pool -> pairs -> locked splits.

SCOPE OF THE CORPUS
-------------------
**Every subject is included on equal footing.** Law, science, BCS and literature
are all in. Nothing is excluded for being hard to answer or for needing outside
knowledge. The only subject missing is `geography`, and not for difficulty: every
geography item was fill-in-the-blank, and this project is QA only (filter 2).

An earlier version of this file filtered on "answerability" and dropped roughly
4,150 pairs across those subjects. That filter has been removed by decision of
the project owner. What replaced it is *measurement*: the `difficulty` field
(see assign_difficulty) marks which pairs a string matcher can solve, so hard
and easy are reported separately instead of one being thrown away.

WHAT IS STILL REMOVED, AND WHY
------------------------------
Only five things, and only one of them (fill-in-the-blank) is about the kind of
item rather than about validity. None is about how hard an item is:

  1. Incomplete pairs      -- a group without both a correct AND a wrong answer
                              cannot form a pair at all.
  2. Fill-in-the-blank     -- THIS PROJECT IS QA ONLY. A cloze item teaches
                              span-copying, not answer checking, and a string
                              matcher scores 0.929 on them. Removing these also
                              removes `geography`, which was 100% cloze.
  3. OCR-damaged text      -- stranded Bengali vowel signs ("বিষয়ের ি").
                              Unreadable, so unlabelable. About 3 pairs.
  4. Question cut verbatim -- a question containing a 6+ word run copied straight
     out of the passage       out of its own passage answers itself.
  5. Duplicate questions   -- the same question text twice would put one copy in
                              train and the other in test. That is leakage.

Composition constraints (PRD D2/D3) still apply: 60/40 has-context to
no-context, 50/50 labels, and a per-subject cap so no single subject dominates a
condition -- mathematics alone supplies 59% of the no-context pool and would
otherwise swamp it.

Run from the repository root:

    python src/build_corpus.py

Deterministic: seed 42, byte-identical output on every rerun.
"""

from __future__ import annotations

import collections
import difflib
import json
import random
import re
from pathlib import Path

INTERIM = Path("data/interim/bn_pool.jsonl")
SCHEMA = Path("configs/schema.json")
CORPUS_DIR = Path("data/corpus/bn_v1")
SPLITS_DIR = Path("data/splits")

SEED = 42
NO_CONTEXT_RATIO = 40 / 60      # PRD D2: 60% has-context / 40% no-context
# No single subject may dominate a condition. The limit differs by condition
# because the two look very different after filtering:
#   no-context has 11 subjects and mathematics alone supplies 59% of the
#     filtered pool, so a strict 30% cap is needed -- and is easily met.
#   has-context has only 5, and the two largest (reading_comprehension and
#     history) are the SAME task -- an answer grounded in a Wikipedia passage --
#     differing only in topic. Forcing them to 30% would discard a quarter of
#     the scarcest data in the project to fix an imbalance that does not
#     correspond to a real difference in what the model has to do.
MAX_SUBJECT_SHARE = {"has_context": 0.50, "no_context": 0.30}
SPLIT_FRACTIONS = (0.70, 0.15, 0.15)   # train / dev / test

# NOTE: there is no subject whitelist. Every subject in the pool is eligible.

# Fill-in-the-blank items are EXCLUDED. This project is question-answering only.
# A cloze item ("____ elakay pradhanata peshadar jele...") is a different task:
# the model learns to copy the missing span out of the passage rather than to
# judge whether an answer is supported. They were also unusually easy for a
# string matcher (0.929 on their own vs 0.839 corpus-wide), so keeping them
# inflated every has-context score. This also removes `geography`, which was
# 100% fill-in-the-blank.
FITB = re.compile(r"শূন্যস্থান|_{3,}|\.{4,}")

# OCR damage: a Bengali combining vowel sign left stranded as its own word
# ("বিষয়ের ি"). Only a handful of pairs, but they are unreadable, so they are
# dropped rather than sent to an annotator.
STRAY_MARK = re.compile(r"(?:^|\s)[়-ৗ]+(?:\s|$)")

# Language questions that were mis-filed under "history" in the source pool.
LANGUAGE_QUESTION = re.compile(r"সন্ধি|বিপরীত|সমার্থক|প্রত্যয়|সমাস|বাগধারা|শুদ্ধ\s*বানান|পদ\s*কোনটি")


def load_complete_pairs() -> list[list[dict]]:
    """Groups the interim pool into pairs, keeping only genuine ones.

    A pair is genuine when it has exactly 2 members AND they carry different
    labels. Some groups in the pool hold two wrong answers and no correct one;
    those cannot form a correct/hallucinated pair and are discarded.
    """
    groups: dict[str, list[dict]] = collections.defaultdict(list)
    for line in INTERIM.open(encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            groups[r["pair_id"]].append(r)
    return [v for v in groups.values()
            if len(v) == 2 and len({m["label"] for m in v}) == 2]


def relabel_subject(pair: list[dict]) -> None:
    """Re-files the grammar questions that the source pool filed under history.

    A block of sandhi-bicched / somartho-shobdo questions sits in the history
    files. Only those are moved -- genuine history questions in the same files
    keep their subject. An earlier version moved every history no-context item
    unconditionally, which was safe only because real history trivia was being
    dropped elsewhere; now that nothing is dropped, the check has to be real.
    """
    if pair[0]["subject"] == "history" and pair[0]["condition"] == "no_context":
        if LANGUAGE_QUESTION.search(pair[0]["question"]):
            for m in pair:
                m["subject"] = "grammar"


def toks(s: str) -> set[str]:
    return set(re.findall(r"\w+", str(s)))


def longest_verbatim_run(question: str, context: str) -> int:
    """Longest run of consecutive question words that appears word-for-word in
    the passage. Used to spot a question that was cut straight out of the text.

    This is deliberately a CONTIGUOUS-RUN check rather than a bag-of-words
    overlap. A real comprehension question is *supposed* to reuse the passage's
    words -- "What was Titumir's father's name?" shares most of its words with
    any passage about Titumir. Scoring that by word overlap rejects 97% of
    perfectly good questions, which is exactly what an earlier version of this
    file did.
    """
    words = re.findall(r"\w+", question)
    best = 0
    for i in range(len(words)):
        for j in range(i + best + 1, len(words) + 1):
            if " ".join(words[i:j]) in context:
                best = max(best, j - i)
            else:
                break
    return best


def passes_content_filters(pair: list[dict]) -> bool:
    """Validity checks. See the module docstring for what each one is for."""
    correct = next(m for m in pair if m["label"] == 1)
    wrong = next(m for m in pair if m["label"] == 0)
    q, ctx = correct["question"], correct["context"]

    # QA only: fill-in-the-blank is a different task (see the FITB note above).
    if FITB.search(q):
        return False

    # Unreadable text cannot be annotated, so it cannot be used.
    if any(STRAY_MARK.search(str(x)) for x in
           (q, correct["candidate_answer"], wrong["candidate_answer"])):
        return False

    if correct["condition"] == "no_context":
        return True

    # -- has-context only from here ------------------------------------------
    if not ctx or ctx == "[NULL]":
        return False

    # A question cut straight out of the passage gives its own answer away.
    if longest_verbatim_run(q, ctx) >= 6:
        return False

    # NOTE: pairs whose WRONG answer is ALSO a verbatim span of the passage are
    # deliberately KEPT. An earlier version of this file dropped them, which was
    # backwards -- dropping them leaves only pairs where "the answer appears in
    # the passage" lines up exactly with "the answer is correct", and that is
    # what pushed the string-matching baseline to 0.855. Those pairs are the
    # shortcut-proof ones; they are kept and marked hard by assign_difficulty().
    return True


def answer_in_context(member: dict) -> bool:
    """Mirrors the context-substring probe in src/audit.py exactly."""
    return str(member["candidate_answer"]).strip().rstrip("।. ") in member["context"]


def assign_difficulty(pair: list[dict]) -> None:
    """Marks a pair easy or hard. The meaning differs by condition.

    has-context -- HARD when the string shortcut does NOT separate the two
      answers: either both are verbatim spans of the passage, or neither is. On
      those pairs "does this text appear in the passage?" carries no information
      about the label, so a model has to actually check whether the passage
      supports the answer. EASY when the shortcut does separate them.

      This is the split worth reporting separately. The substring baseline
      scores 0.812 across all has-context records, 0.980 on easy and 0.456 on
      hard, so a has-context score means little without both numbers.

    no-context -- there is no passage, so the substring rule cannot apply at
      all. HARD here means the wrong answer is a near-miss: close in surface
      form to the correct one (off-by-one number, one name part swapped) rather
      than obviously unrelated.
    """
    correct = next(m for m in pair if m["label"] == 1)
    wrong = next(m for m in pair if m["label"] == 0)

    if correct["condition"] == "has_context":
        hard = answer_in_context(correct) == answer_in_context(wrong)
    else:
        hard = difflib.SequenceMatcher(
            None, str(correct["candidate_answer"]), str(wrong["candidate_answer"])
        ).ratio() >= 0.60

    for m in pair:
        m["difficulty"] = "hard" if hard else "easy"


def hardness(pair: list[dict]) -> float:
    """Higher = the wrong answer reuses more of the passage's vocabulary, so it
    cannot be rejected by checking whether its words appear in the passage. These
    are the informative negatives; they are preferred when selecting."""
    wrong = next(m for m in pair if m["label"] == 0)
    ct = toks(wrong["context"])
    wt = toks(wrong["candidate_answer"])
    if not wt or not ct:
        return 0.0
    return len(wt & ct) / len(wt)


def apply_subject_cap(pairs: list[list[dict]], n: int, cap_share: float,
                      rng: random.Random) -> list[list[dict]]:
    """Takes up to n pairs, letting no subject exceed cap_share of the result.

    The cap is a share of the FINAL total, not of the requested n. Those differ
    whenever a cap actually binds -- capping the big subjects shrinks the total,
    which shrinks the cap again. This solves for the fixed point instead of
    applying the cap once and overshooting.
    """
    by_subject: dict[str, list] = collections.defaultdict(list)
    for p in pairs:
        by_subject[p[0]["subject"]].append(p)
    for v in by_subject.values():
        rng.shuffle(v)

    total = min(n, len(pairs))
    for _ in range(50):
        cap = max(1, int(total * cap_share))
        achieved = min(n, sum(min(len(v), cap) for v in by_subject.values()))
        if achieved == total:
            break
        total = achieved
    cap = max(1, int(total * cap_share))
    n = total

    chosen: list[list[dict]] = []
    # Round-robin across subjects so smaller subjects are not starved.
    while len(chosen) < n:
        added = False
        for subject in sorted(by_subject):
            bucket = by_subject[subject]
            taken = sum(1 for c in chosen if c[0]["subject"] == subject)
            if bucket and taken < cap and len(chosen) < n:
                chosen.append(bucket.pop())
                added = True
        if not added:
            break
    return chosen


def make_splits(pairs: list[list[dict]], rng: random.Random) -> dict[str, list[list[dict]]]:
    """Splits BY PAIR so a question and both its answers never straddle a split.

    Splitting by record instead would put the correct answer in train and the
    hallucinated answer in test, letting a model score by memorising the
    question rather than by judging the answer.
    """
    buckets: dict[tuple, list] = collections.defaultdict(list)
    for p in pairs:
        buckets[(p[0]["condition"], p[0]["subject"])].append(p)

    out = {"train": [], "dev": [], "test": []}
    for key in sorted(buckets):
        group = buckets[key]
        rng.shuffle(group)
        n = len(group)
        n_train = int(n * SPLIT_FRACTIONS[0])
        n_dev = int(n * (SPLIT_FRACTIONS[0] + SPLIT_FRACTIONS[1])) - n_train
        out["train"] += group[:n_train]
        out["dev"] += group[n_train:n_train + n_dev]
        out["test"] += group[n_train + n_dev:]
    for k in out:
        rng.shuffle(out[k])
    return out


def validate(records: list[dict], schema_path: Path) -> list[str]:
    schema = json.loads(schema_path.read_text(encoding="utf-8"))
    required = schema["required"]
    enums = {k: v["enum"] for k, v in schema["properties"].items() if "enum" in v}
    problems: list[str] = []
    for r in records:
        for f in required:
            if f not in r:
                problems.append(f"{r.get('id')}: missing {f}")
            elif r[f] in (None, "") and not (f == "context" and r["condition"] == "no_context"):
                problems.append(f"{r.get('id')}: empty {f}")
        for f, allowed in enums.items():
            if f in r and r[f] not in allowed:
                problems.append(f"{r.get('id')}: {f}={r[f]!r} not in enum")
        if r["label"] == 1 and r["hallucination_type"] != "none":
            problems.append(f"{r['id']}: label=1 but type={r['hallucination_type']}")
        if r["label"] == 0 and r["hallucination_type"] == "none":
            problems.append(f"{r['id']}: label=0 but type=none")
    return problems


def write_jsonl(path: Path, pairs: list[list[dict]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for p in pairs:
            for m in sorted(p, key=lambda x: -x["label"]):
                fh.write(json.dumps(m, ensure_ascii=False, sort_keys=True) + "\n")


def main() -> None:
    rng = random.Random(SEED)
    all_pairs = load_complete_pairs()
    print(f"complete pairs in interim pool          : {len(all_pairs):,}")

    for p in all_pairs:
        relabel_subject(p)

    # Every subject is eligible; only the validity guards in the module
    # docstring remove anything.
    kept = [p for p in all_pairs if passes_content_filters(p)]
    print(f"after validity filters                  : {len(kept):,}")

    # Two different source rows sometimes carry identical question text. Keeping
    # both would place one copy in train and the other in test -- leakage that
    # lets a model score by recall instead of by judging the answer.
    seen_q: set[str] = set()
    deduped = []
    for p in sorted(kept, key=lambda x: x[0]["id"]):
        q = " ".join(p[0]["question"].split())
        if q in seen_q:
            continue
        seen_q.add(q)
        deduped.append(p)
    print(f"after dropping repeated questions       : {len(deduped):,}")
    kept = deduped

    has_ctx = [p for p in kept if p[0]["condition"] == "has_context"]
    no_ctx = [p for p in kept if p[0]["condition"] == "no_context"]
    print(f"   has-context available                : {len(has_ctx):,}")
    print(f"   no-context available                 : {len(no_ctx):,}")

    # Stage 3 -- prefer hard negatives in has-context, then cap subjects.
    has_ctx.sort(key=hardness, reverse=True)
    has_selected = apply_subject_cap(has_ctx, len(has_ctx),
                                     MAX_SUBJECT_SHARE["has_context"], rng)

    n_no_ctx = int(round(len(has_selected) * NO_CONTEXT_RATIO))
    no_selected = apply_subject_cap(no_ctx, min(n_no_ctx, len(no_ctx)),
                                    MAX_SUBJECT_SHARE["no_context"], rng)

    corpus = has_selected + no_selected
    for p_ in corpus:
        assign_difficulty(p_)
    rng.shuffle(corpus)
    records = [m for p in corpus for m in p]

    problems = validate(records, SCHEMA)
    if problems:
        print(f"\n*** {len(problems)} VALIDATION PROBLEMS -- nothing written ***")
        for x in problems[:10]:
            print("   ", x)
        raise SystemExit(1)

    write_jsonl(CORPUS_DIR / "corpus.jsonl", corpus)
    splits = make_splits(corpus, rng)
    for name, pairs in splits.items():
        write_jsonl(SPLITS_DIR / f"{name}.jsonl", pairs)

    # -- report --------------------------------------------------------------
    print(f"\nCORPUS: {len(corpus):,} pairs / {len(records):,} records -> {CORPUS_DIR}/corpus.jsonl")
    hc = sum(1 for p in corpus if p[0]["condition"] == "has_context")
    print(f"  has-context {hc:,} ({hc/len(corpus):.0%})   no-context {len(corpus)-hc:,} ({1-hc/len(corpus):.0%})")
    print(f"  labels: {sum(1 for r in records if r['label']==1):,} correct / "
          f"{sum(1 for r in records if r['label']==0):,} hallucinated")

    print("\n  subject mix:")
    for cond in ("has_context", "no_context"):
        sub = collections.Counter(p[0]["subject"] for p in corpus if p[0]["condition"] == cond)
        total = sum(sub.values())
        print(f"    {cond}:")
        for s, n in sub.most_common():
            print(f"      {s:<24}{n:>6,}  {n/total:5.1%}")

    print("")
    fitb = sum(1 for x in corpus if FITB.search(x[0]["question"]))
    assert fitb == 0, f"{fitb} fill-in-the-blank items leaked into the corpus"
    print("")
    print("  fill-in-the-blank items: 0  (QA only -- see the FITB note in this file)")
    print("  difficulty (hard = the string shortcut does not work on this pair):")
    for cond in ("has_context", "no_context"):
        d = collections.Counter(x[0]["difficulty"] for x in corpus if x[0]["condition"] == cond)
        tot = sum(d.values())
        print(f"    {cond:<13} hard {d['hard']:>5,} ({d['hard']/tot:4.0%})   "
              f"easy {d['easy']:>5,} ({d['easy']/tot:4.0%})")
    print("\nSPLITS (grouped by pair -- a pair never straddles two splits):")
    for name in ("train", "dev", "test"):
        pairs = splits[name]
        h = sum(1 for p in pairs if p[0]["condition"] == "has_context")
        print(f"  {name:<6}{len(pairs):>6,} pairs  {len(pairs)*2:>6,} records   has-ctx {h/len(pairs):5.1%}")

    # Leakage guard: the same question must not appear in two splits.
    seen: dict[str, str] = {}
    leaks = 0
    for name in ("train", "dev", "test"):
        for p in splits[name]:
            q = p[0]["question"]
            if q in seen and seen[q] != name:
                leaks += 1
            seen[q] = name
    print(f"\n  duplicate questions across splits: {leaks}  {'OK' if leaks == 0 else '*** LEAKAGE ***'}")


if __name__ == "__main__":
    main()
