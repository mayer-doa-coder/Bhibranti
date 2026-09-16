"""LLM-assisted annotation for M3, with cross-validation between judges.

WHAT THIS DOES, AND WHAT IT DELIBERATELY DOES NOT DO
----------------------------------------------------
The corpus already carries a binary label (1 = correct, 0 = hallucinated) for
all 8,960 records. Annotation therefore has two jobs, and this script treats
them very differently:

  TYPE task  -- fill `hallucination_type` on the 4,480 wrong answers (PRD D8).
                This is reporting metadata. It is never trained on and never
                predicted, so an LLM proposing it carries little risk.
                A deterministic rule already covers 61% of them objectively
                (a changed number, or a plainly different closed-book answer);
                only the remaining ~1,761 are sent to a judge.

  VERIFY task -- ask judges whether they agree with the EXISTING binary label.
                Judges never overwrite it. Where they disagree, the row is
                flagged for a human. This is quality control, not labelling.

    ==> The binary label is the thing the detector predicts. Letting an LLM
        write it would make the corpus circular: the detector would be trained
        to reproduce the judge's blind spots, and a hallucination detector that
        has learned an LLM's hallucinations is measuring nothing. So this
        script cannot and does not write it.

  ==> PRD D4 ("test split 100% human-verified, double-annotated") is NOT
      satisfied by this script. LLM agreement is not human verification. The
      test split still needs two people. Running this does not change that,
      and the report it writes says so.

CROSS-VALIDATION
----------------
Every item goes to N independent judges from different providers. A proposal is
ACCEPTED only when at least `--threshold` of them agree; otherwise it is FLAGGED
for a human. Disagreement is the useful signal here -- it concentrates human
effort on the items that actually need judgement.

PROVIDERS (free tiers first)
----------------------------
  gemini      GEMINI_API_KEY      free tier, aistudio.google.com/apikey
  groq        GROQ_API_KEY        free tier, console.groq.com/keys
  openrouter  OPENROUTER_API_KEY  has :free models, openrouter.ai/keys
  anthropic   ANTHROPIC_API_KEY   paid
  openai      OPENAI_API_KEY      paid

Anthropic and OpenAI have no free tier; use gemini + groq for a no-cost run.
Two providers is the minimum for cross-validation; three is better.

USAGE
-----
    # prove the pipeline works offline, no keys, no cost
    python src/llm_annotate.py --dry-run --limit 60

    # real run once at least two keys are set
    python src/llm_annotate.py --judges gemini,groq --task both

    # resume: cached responses are never re-requested
    python src/llm_annotate.py --judges gemini,groq --task both

Outputs go to data/annotated/llm_round1/ and NOTHING is written back into the
corpus. Use src/merge_annotation.py for that, after the flagged rows are
reviewed.
"""

from __future__ import annotations

import argparse
import collections
import difflib
import hashlib
import json
import os
import random
import re
import sys
import time
from pathlib import Path

import requests

CORPUS = Path("data/corpus/bn_v1/corpus.jsonl")
OUT_DIR = Path("data/annotated/llm_round1")
CACHE = OUT_DIR / "_cache.jsonl"

INTRINSIC = ["entity", "numeric", "relational", "contradiction"]
EXTRINSIC = ["fabricated", "overclaim"]

DIGITS = re.compile(r"[\d০-৯]+")

# ---------------------------------------------------------------------------
# Providers. Plain REST so no SDK has to be installed.
# ---------------------------------------------------------------------------
PROVIDERS = {
    "gemini": {
        "env": "GEMINI_API_KEY",
        "model": "gemini-2.0-flash",
        "free": True,
        "rpm": 15,
    },
    "groq": {
        "env": "GROQ_API_KEY",
        "model": "llama-3.3-70b-versatile",
        "free": True,
        "rpm": 30,
    },
    "openrouter": {
        "env": "OPENROUTER_API_KEY",
        "model": "meta-llama/llama-3.3-70b-instruct:free",
        "free": True,
        "rpm": 20,
    },
    "anthropic": {
        "env": "ANTHROPIC_API_KEY",
        "model": "claude-sonnet-5",
        "free": False,
        "rpm": 50,
    },
    "openai": {
        "env": "OPENAI_API_KEY",
        "model": "gpt-4.1-mini",
        "free": False,
        "rpm": 50,
    },
}


def call_provider(name: str, prompt: str, timeout: int = 60) -> str:
    """One request, one raw text reply. Raises on transport failure."""
    cfg = PROVIDERS[name]
    key = os.environ.get(cfg["env"], "")
    model = os.environ.get(f"{name.upper()}_MODEL", cfg["model"])

    if name == "gemini":
        url = (f"https://generativelanguage.googleapis.com/v1beta/models/"
               f"{model}:generateContent?key={key}")
        body = {
            "contents": [{"parts": [{"text": prompt}]}],
            "generationConfig": {"temperature": 0, "responseMimeType": "application/json"},
        }
        r = requests.post(url, json=body, timeout=timeout)
        r.raise_for_status()
        return r.json()["candidates"][0]["content"]["parts"][0]["text"]

    if name == "anthropic":
        r = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers={"x-api-key": key, "anthropic-version": "2023-06-01",
                     "content-type": "application/json"},
            json={"model": model, "max_tokens": 300, "temperature": 0,
                  "messages": [{"role": "user", "content": prompt}]},
            timeout=timeout)
        r.raise_for_status()
        return r.json()["content"][0]["text"]

    # groq / openrouter / openai are all OpenAI-compatible
    url = {
        "groq": "https://api.groq.com/openai/v1/chat/completions",
        "openrouter": "https://openrouter.ai/api/v1/chat/completions",
        "openai": "https://api.openai.com/v1/chat/completions",
    }[name]
    r = requests.post(
        url,
        headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
        json={"model": model, "temperature": 0, "max_tokens": 300,
              "response_format": {"type": "json_object"},
              "messages": [{"role": "user", "content": prompt}]},
        timeout=timeout)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


# ---------------------------------------------------------------------------
# Prompts. These mirror docs/ANNOTATION_GUIDELINES.md so a judge applies the
# same rules a human annotator would, including the rules that were added
# after the first agreement round.
# ---------------------------------------------------------------------------
RULES = """You are annotating a Bengali question-answering dataset for hallucination
detection. Apply these rules exactly; they are the same rules the human
annotators follow.

If a PASSAGE is given, judge ONLY against the passage. Do not use outside
knowledge. An answer that is true in the world but not supported by the passage
is WRONG. An answer that says the same thing in different words IS supported.

If NO passage is given, judge whether the answer is true in general.

Other rules:
- A partly correct answer is wrong.
- Ignore hedges such as "probably".
- A near-miss number or name is wrong, not correct.
- Spelling variation alone does not make an answer wrong, UNLESS the question is
  about the form of the word (sandhi, samas, spelling).
- Ignore citation markers like [1][2].
- If the answer repeats a whole passage sentence but contains the asked-for
  fact, it is correct."""

TYPE_TASK = """{rules}

TASK: this answer is already known to be WRONG. Choose the single best category
for WHY it is wrong.

Allowed categories for this item: {allowed}

  entity        - wrong person, place, or organisation
  numeric       - wrong number, date, quantity or unit
  relational    - right entities but the relationship between them is scrambled
  contradiction - states the opposite of the passage
  fabricated    - simply not true; invented
  overclaim     - answers confidently something that has no fixed answer

{item}

Reply with JSON only: {{"type": "<one of {allowed}>", "confidence": <0-1>}}"""

VERIFY_TASK = """{rules}

TASK: decide whether the candidate answer is CORRECT or WRONG.

{item}

Reply with JSON only: {{"verdict": "correct" or "wrong", "confidence": <0-1>}}"""


def render_item(rec: dict) -> str:
    parts = []
    if rec["condition"] == "has_context" and rec.get("context"):
        parts.append(f"PASSAGE:\n{rec['context']}")
    else:
        parts.append("PASSAGE: (none - judge from general knowledge)")
    parts.append(f"QUESTION:\n{rec['question']}")
    parts.append(f"CANDIDATE ANSWER:\n{rec['candidate_answer']}")
    return "\n\n".join(parts)


def build_prompt(task: str, rec: dict) -> str:
    if task == "type":
        allowed = INTRINSIC if rec["condition"] == "has_context" else EXTRINSIC
        return TYPE_TASK.format(rules=RULES, allowed="|".join(allowed),
                                item=render_item(rec))
    return VERIFY_TASK.format(rules=RULES, item=render_item(rec))


def parse_reply(text: str, task: str, condition: str) -> tuple[str | None, float]:
    """Pull the verdict out of a reply, tolerating markdown fences and prose."""
    m = re.search(r"\{.*\}", text, re.S)
    if m:
        try:
            d = json.loads(m.group(0))
            val = str(d.get("type" if task == "type" else "verdict", "")).strip().lower()
            conf = float(d.get("confidence", 0.5))
            allowed = ((INTRINSIC if condition == "has_context" else EXTRINSIC)
                       if task == "type" else ["correct", "wrong"])
            if val in allowed:
                return val, max(0.0, min(1.0, conf))
        except (ValueError, TypeError):
            pass
    # fall back to a bare keyword anywhere in the reply
    allowed = ((INTRINSIC if condition == "has_context" else EXTRINSIC)
               if task == "type" else ["correct", "wrong"])
    for a in allowed:
        if re.search(rf"\b{a}\b", text, re.I):
            return a, 0.4
    return None, 0.0


# ---------------------------------------------------------------------------
# Deterministic pre-pass: fills the objective cases so judges see only the
# genuinely ambiguous items. Same logic as build_annotation_sheets.py.
# ---------------------------------------------------------------------------
def derive_type_rule(correct: str, wrong: str, condition: str) -> str:
    ca = " ".join(str(correct).split())
    wa = " ".join(str(wrong).split())
    if condition == "has_context":
        if (DIGITS.findall(ca) or DIGITS.findall(wa)) and DIGITS.findall(ca) != DIGITS.findall(wa):
            return "numeric"
        return ""
    if difflib.SequenceMatcher(None, ca, wa).ratio() < 0.90:
        return "fabricated"
    return ""


# ---------------------------------------------------------------------------
# Mock judge for --dry-run: deterministic, so the consensus logic is testable
# offline without keys or cost.
# ---------------------------------------------------------------------------
def mock_reply(judge: str, task: str, rec: dict, agree_rate: float) -> str:
    seed = int(hashlib.sha256(f"{judge}|{rec['id']}|{task}".encode()).hexdigest()[:8], 16)
    rng = random.Random(seed)
    if task == "verify":
        v = "correct" if rec["label"] == 1 else "wrong"
        if rng.random() > agree_rate:
            v = "wrong" if v == "correct" else "correct"
        return json.dumps({"verdict": v, "confidence": round(rng.uniform(0.6, 0.99), 2)})
    allowed = INTRINSIC if rec["condition"] == "has_context" else EXTRINSIC
    pick = allowed[0] if rng.random() < agree_rate else rng.choice(allowed)
    return json.dumps({"type": pick, "confidence": round(rng.uniform(0.5, 0.95), 2)})


# ---------------------------------------------------------------------------
class Cache:
    """Disk cache so a rerun costs nothing and a crash loses nothing."""

    def __init__(self, path: Path):
        self.path = path
        self.data: dict[str, str] = {}
        if path.exists():
            with path.open(encoding="utf-8") as fh:
                for line in fh:
                    if line.strip():
                        d = json.loads(line)
                        self.data[d["k"]] = d["v"]

    @staticmethod
    def key(judge: str, task: str, item_id: str) -> str:
        return f"{judge}|{task}|{item_id}"

    def get(self, k): return self.data.get(k)

    def put(self, k, v):
        self.data[k] = v
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps({"k": k, "v": v}, ensure_ascii=False) + "\n")


def ask(judge: str, task: str, rec: dict, cache: Cache, args) -> tuple[str | None, float]:
    k = Cache.key(judge, task, rec["id"])
    raw = cache.get(k)
    if raw is None:
        if args.dry_run:
            raw = mock_reply(judge, task, rec, args.mock_agree)
        else:
            delay = 60.0 / PROVIDERS[judge]["rpm"]
            last = None
            for attempt in range(4):
                try:
                    raw = call_provider(judge, build_prompt(task, rec))
                    break
                except Exception as e:                      # noqa: BLE001
                    last = e
                    time.sleep(delay * (2 ** attempt))
            else:
                print(f"    [!] {judge} failed on {rec['id']}: {last}")
                return None, 0.0
            time.sleep(delay)
        cache.put(k, raw)
    return parse_reply(raw, task, rec["condition"])


def consensus(votes: dict[str, str | None], threshold: float):
    """Returns (value, agreement_fraction, accepted)."""
    valid = [v for v in votes.values() if v]
    if not valid:
        return None, 0.0, False
    top, n = collections.Counter(valid).most_common(1)[0]
    frac = n / len(votes)
    # Epsilon matters: with 3 judges, 2 agreeing gives 0.6666..., which is
    # LESS than a 0.67 threshold. Without this the default silently demanded
    # unanimity and flagged every 2-of-3 agreement.
    return top, frac, frac >= threshold - 1e-9


# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--judges", default="", help="comma list, e.g. gemini,groq")
    ap.add_argument("--task", choices=["type", "verify", "both"], default="both")
    ap.add_argument("--threshold", type=float, default=2 / 3,
                    help="fraction of judges that must agree (default 2/3, "
                         "i.e. 2 of 3 judges; use 1.0 to demand unanimity)")
    ap.add_argument("--verify-sample", type=int, default=400,
                    help="how many records to spot-check the binary label on")
    ap.add_argument("--limit", type=int, default=0, help="cap items, for testing")
    ap.add_argument("--dry-run", action="store_true",
                    help="use deterministic mock judges; no keys, no cost")
    ap.add_argument("--mock-agree", type=float, default=0.85,
                    help="dry-run only: how often a mock judge agrees")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    # -- pick judges --------------------------------------------------------
    if args.judges:
        judges = [j.strip() for j in args.judges.split(",") if j.strip()]
    elif args.dry_run:
        judges = ["mockA", "mockB", "mockC"]
    else:
        judges = [n for n, c in PROVIDERS.items() if os.environ.get(c["env"])]

    if args.dry_run and not args.judges:
        pass
    else:
        for j in judges:
            if j not in PROVIDERS:
                raise SystemExit(f"unknown judge {j!r}; choose from {list(PROVIDERS)}")
            if not args.dry_run and not os.environ.get(PROVIDERS[j]["env"]):
                raise SystemExit(f"{PROVIDERS[j]['env']} is not set for judge {j!r}")

    if len(judges) < 2:
        raise SystemExit(
            "cross-validation needs at least 2 judges.\n"
            "Set two of: GEMINI_API_KEY (free), GROQ_API_KEY (free), "
            "OPENROUTER_API_KEY (free), ANTHROPIC_API_KEY, OPENAI_API_KEY\n"
            "Or try the pipeline offline with:  --dry-run")

    print(f"judges: {judges}   threshold: {args.threshold:.2f}   "
          f"{'DRY RUN (mock judges)' if args.dry_run else 'LIVE'}")

    # -- load corpus, group into pairs --------------------------------------
    rows = [json.loads(l) for l in CORPUS.open(encoding="utf-8") if l.strip()]
    by_pair: dict[str, list[dict]] = collections.defaultdict(list)
    for r in rows:
        by_pair[r["pair_id"]].append(r)

    # -- TYPE: rule first, judges only for the remainder --------------------
    rule_filled: dict[str, str] = {}
    need_judge: list[dict] = []
    for pid in sorted(by_pair):
        members = by_pair[pid]
        if len(members) != 2:
            continue
        c = next((m for m in members if m["label"] == 1), None)
        w = next((m for m in members if m["label"] == 0), None)
        if not c or not w:
            continue
        t = derive_type_rule(c["candidate_answer"], w["candidate_answer"], w["condition"])
        if t:
            rule_filled[w["id"]] = t
        else:
            need_judge.append(w)

    if args.limit:
        need_judge = need_judge[:args.limit]

    print(f"\nTYPE task: {len(rule_filled):,} filled by rule, "
          f"{len(need_judge):,} sent to judges")

    cache = Cache(CACHE)
    proposals: list[dict] = []

    if args.task in ("type", "both"):
        for i, rec in enumerate(need_judge, 1):
            votes, confs = {}, {}
            for j in judges:
                v, conf = ask(j, "type", rec, cache, args)
                votes[j], confs[j] = v, conf
            val, frac, ok = consensus(votes, args.threshold)
            proposals.append({
                "task": "type", "id": rec["id"], "pair_id": rec["pair_id"],
                "subject": rec["subject"], "condition": rec["condition"],
                "votes": votes, "confidence": confs,
                "proposed": val, "agreement": round(frac, 3), "accepted": ok,
            })
            if i % 100 == 0 or i == len(need_judge):
                print(f"  type {i}/{len(need_judge)}")

    # -- VERIFY: does a judge agree with the label already in the corpus? ----
    if args.task in ("verify", "both"):
        rng = random.Random(args.seed)
        sample = rows[:]
        rng.shuffle(sample)
        sample = sample[:args.limit or args.verify_sample]
        print(f"\nVERIFY task: spot-checking {len(sample):,} existing labels")
        for i, rec in enumerate(sample, 1):
            votes, confs = {}, {}
            for j in judges:
                v, conf = ask(j, "verify", rec, cache, args)
                votes[j], confs[j] = v, conf
            val, frac, ok = consensus(votes, args.threshold)
            existing = "correct" if rec["label"] == 1 else "wrong"
            proposals.append({
                "task": "verify", "id": rec["id"], "pair_id": rec["pair_id"],
                "subject": rec["subject"], "condition": rec["condition"],
                "votes": votes, "confidence": confs,
                "proposed": val, "agreement": round(frac, 3),
                "existing_label": existing,
                "judges_agree_with_corpus": (val == existing) if val else None,
                "accepted": ok and val == existing,
            })
            if i % 100 == 0 or i == len(sample):
                print(f"  verify {i}/{len(sample)}")

    # -- write outputs ------------------------------------------------------
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with (OUT_DIR / "rule_types.json").open("w", encoding="utf-8") as fh:
        json.dump(rule_filled, fh, ensure_ascii=False, indent=1, sort_keys=True)

    with (OUT_DIR / "proposals.jsonl").open("w", encoding="utf-8") as fh:
        for p in proposals:
            fh.write(json.dumps(p, ensure_ascii=False, sort_keys=True) + "\n")

    import csv
    corpus_by_id = {r["id"]: r for r in rows}
    flagged = [p for p in proposals if not p["accepted"]]
    with (OUT_DIR / "flagged_for_human.csv").open("w", encoding="utf-8-sig", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["item_id", "task", "why", "subject", "condition",
                    "question", "answer", "judge_votes", "agreement",
                    "your_decision", "your_notes"])
        for p in flagged:
            rec = corpus_by_id[p["id"]]
            if p["task"] == "verify":
                why = ("judges disagree with the corpus label"
                       if p.get("judges_agree_with_corpus") is False
                       else "judges disagree with each other")
            else:
                why = "judges disagree on the error type"
            w.writerow([p["id"], p["task"], why, p["subject"], p["condition"],
                        " ".join(rec["question"].split())[:300],
                        " ".join(str(rec["candidate_answer"]).split())[:300],
                        "; ".join(f"{k}={v}" for k, v in p["votes"].items()),
                        p["agreement"], "", ""])

    # -- report -------------------------------------------------------------
    t_props = [p for p in proposals if p["task"] == "type"]
    v_props = [p for p in proposals if p["task"] == "verify"]
    t_ok = sum(1 for p in t_props if p["accepted"])
    v_ok = sum(1 for p in v_props if p["accepted"])
    v_conflict = sum(1 for p in v_props if p.get("judges_agree_with_corpus") is False)

    print("\n" + "=" * 64)
    print(f"  types filled by rule        : {len(rule_filled):,}")
    if t_props:
        print(f"  types accepted by consensus : {t_ok:,}/{len(t_props):,} "
              f"({t_ok/len(t_props):.0%})")
    if v_props:
        print(f"  labels confirmed by judges  : {v_ok:,}/{len(v_props):,} "
              f"({v_ok/len(v_props):.0%})")
        print(f"  labels judges DISPUTE       : {v_conflict:,}  <- read these first")
    print(f"  flagged for a human         : {len(flagged):,}")
    print(f"\n  outputs in {OUT_DIR}/")
    print("  NOTHING was written into the corpus. Review the flagged file, then")
    print("  run src/merge_annotation.py.")
    print("=" * 64)


if __name__ == "__main__":
    main()
