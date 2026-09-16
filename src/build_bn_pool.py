"""Normalise the Bengali-script QA pool into the BanglishHallu record schema.

Reads the raw vendor files in ``data/raw/bn_qa_pool/`` and produces:

  * ``data/interim/bn_pool.jsonl``            — full cleaned pool, schema-mapped
  * ``data/splits/_pool_sanity/*.jsonl``  — throwaway grouped splits used only to
    sanity-check this script. The REAL Phase 1 splits are built by
    ``src/build_corpus.py``, which pairs, filters and splits the pool.

SOURCES — the content comes from Bengali Wikipedia (CC BY-SA 4.0) and BCS question
banks; the QA pairs were built from those texts with LLM assistance. Full register
and licence constraints in data/SOURCES.md.

IMPORTANT — this pool is **monolingual Bengali script**, not Banglish. It is the
base QA layer that the Banglish condition is produced *from* (see
docs/IMPLEMENTATION_GUIDE.md section 3.3), not the Phase 1 corpus itself. The splits it
produces are a pipeline-development benchmark and are explicitly NOT the locked
Phase 1 test set. See data/DATASET_AUDIT.md.

LABEL CONVENTION (PRD section 5.1a) — ``1 = correct/faithful``, ``0 = incorrect/
hallucinated``. This matches the vendor files' native polarity, so **no flip is
applied**. The label is copied through unchanged.

One correction is applied to the vendor data: exact duplicates and contradictory
records (identical text carrying both labels) are dropped.

Run from the repository root:

    python src/build_bn_pool.py
"""

from __future__ import annotations

import collections
import hashlib
import json
import random
import unicodedata
from pathlib import Path

RAW_DIR = Path("data/raw/bn_qa_pool")
INTERIM = Path("data/interim/bn_pool.jsonl")
SPLIT_DIR = Path("data/splits/_pool_sanity")

SEED = 42
NULL_CONTEXT = "[NULL]"

# Benchmark composition, mirroring the PRD section 5.1 corpus targets so that pipeline
# numbers are directly comparable to what the real corpus will produce.
TARGET_RECORDS = 4_000
HAS_CONTEXT_SHARE = 0.60
SPLIT_SIZES = {"train": 3_000, "dev": 500, "test": 500}

# No single subject may exceed this share of either condition. The raw pool is
# 52% mathematics; without a cap the benchmark measures arithmetic, not
# hallucination detection.
MAX_SUBJECT_SHARE = 0.25


def subject_of(path: Path) -> str:
    return path.stem.replace("small_", "").replace("_training_data", "")


def bengali_ratio(text: str) -> float:
    """Share of alphabetic characters written in the Bengali block."""
    bn = sum(1 for c in text if "ঀ" <= c <= "৿")
    latin = sum(1 for c in text if c.isascii() and c.isalpha())
    total = bn + latin
    return bn / total if total else 0.0


def proxy_cmi(text: str) -> float:
    """Code-Mixing Index (Das & Gambaeck) using script as the language proxy.

    A script proxy is only valid because this pool has no romanised Bangla: a
    Latin-script token here is genuinely English. Once the Banglish condition
    exists this must be replaced by real token-level language ID, because
    romanised Bangla is Latin script but is *not* English.

    Returns 0.0 for monolingual text, rising toward 50.0 for even mixing.
    """
    tokens = text.split()
    counts = collections.Counter()
    language_independent = 0
    for tok in tokens:
        letters = [c for c in tok if c.isalpha()]
        if not letters:
            language_independent += 1
            continue
        bn = sum(1 for c in letters if "ঀ" <= c <= "৿")
        counts["bn" if bn > len(letters) / 2 else "en"] += 1
    n = len(tokens) - language_independent
    if n <= 0 or not counts:
        return 0.0
    return round(100 * (1 - max(counts.values()) / n), 2)


def load_raw() -> list[dict]:
    """Read the vendor files, stamping provenance. Labels pass through unchanged."""
    records: list[dict] = []
    for path in sorted(RAW_DIR.glob("*.jsonl")):
        subject = subject_of(path)
        with path.open(encoding="utf-8") as fh:
            for lineno, line in enumerate(fh, start=1):
                line = line.strip()
                if not line:
                    continue
                raw = json.loads(line)
                records.append(
                    {
                        "_subject": subject,
                        "_source_file": path.name,
                        "_source_line": lineno,
                        "context": raw["context"],
                        "question": unicodedata.normalize("NFC", raw["prompt_bn"]),
                        "answer": unicodedata.normalize("NFC", str(raw["response_bn"])),
                        # No flip: vendor polarity (1=correct) already matches
                        # the project convention (PRD section 5.1a).
                        "label": int(raw["label"]),
                    }
                )
    return records


def clean(records: list[dict]) -> tuple[list[dict], dict[str, int]]:
    """Drop exact duplicates and text that carries both labels."""
    stats = {"input": len(records)}

    seen: set[tuple] = set()
    deduped = []
    for r in records:
        key = (r["context"], r["question"], r["answer"], r["label"])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(r)
    stats["exact_duplicates_dropped"] = len(records) - len(deduped)

    labels_per_text = collections.defaultdict(set)
    for r in deduped:
        labels_per_text[(r["context"], r["question"], r["answer"])].add(r["label"])
    contradictory = {k for k, v in labels_per_text.items() if len(v) > 1}
    kept = [
        r
        for r in deduped
        if (r["context"], r["question"], r["answer"]) not in contradictory
    ]
    stats["contradictory_dropped"] = len(deduped) - len(kept)

    non_empty = [r for r in kept if r["answer"].strip() and r["question"].strip()]
    stats["empty_dropped"] = len(kept) - len(non_empty)
    stats["output"] = len(non_empty)
    return non_empty, stats


def to_schema(records: list[dict]) -> list[dict]:
    """Map onto the frozen record schema, grouping records into label pairs."""
    groups: dict[tuple, list[dict]] = collections.defaultdict(list)
    for r in records:
        groups[(r["context"], r["question"])].append(r)

    # Deterministic ids: sort by a stable content hash, not by dict order.
    ordered = sorted(
        groups.items(),
        key=lambda kv: hashlib.sha256(
            (kv[0][0] + "␟" + kv[0][1]).encode("utf-8")
        ).hexdigest(),
    )

    out: list[dict] = []
    for pair_index, ((context, question), members) in enumerate(ordered):
        pair_id = f"bnp_{pair_index:06d}"
        has_context = context != NULL_CONTEXT
        complete = len({m["label"] for m in members}) == 2
        # Sort faithful (1) first so member _0 is always the correct answer.
        for member_index, r in enumerate(sorted(members, key=lambda m: -m["label"])):
            text = (context if has_context else "") + " " + question + " " + r["answer"]
            out.append(
                {
                    "id": f"{pair_id}_{member_index}",
                    "pair_id": pair_id,
                    "pair_complete": complete,
                    "source": "bn_qa_pool",
                    "source_corpus": "bengali_wikipedia+bcs_question_banks",
                    "source_file": r["_source_file"],
                    "source_line": r["_source_line"],
                    "subject": r["_subject"],
                    "level": "unknown",
                    "condition": "has_context" if has_context else "no_context",
                    "context": context if has_context else "",
                    "question": question,
                    # The vendor pool has no separate gold answer. For a complete
                    # pair the faithful member serves as the reference; otherwise
                    # there is none and the field stays empty.
                    "reference_answer": next(
                        (m["answer"] for m in members if m["label"] == 1), ""
                    ),
                    "candidate_answer": r["answer"],
                    "label": r["label"],
                    # Not human-annotated. Provisional value; never ship as final.
                    # none <-> label == 1 (correct); see PRD section 5.1a.
                    "hallucination_type": "none" if r["label"] == 1 else "unlabeled",
                    "difficulty": "unlabeled",
                    # QA pairs were built with LLM assistance (Claude, version not yet
                    # recorded) from Wikipedia passages and BCS banks. See data/SOURCES.md.
                    "generator_model": "claude_version_unrecorded",
                    "generation_seed": None,
                    "annotator_1": None,
                    "annotator_2": None,
                    "adjudicated": False,
                    "provenance": "llm_generated",
                    "script_condition": "bengali_script",
                    "cmi": proxy_cmi(text),
                    "cmi_method": "script_proxy",
                    "bengali_ratio": round(bengali_ratio(text), 4),
                    "error_span": "",
                }
            )
    return out


def build_benchmark(pool: list[dict]) -> dict[str, list[dict]]:
    """Sample a subject-capped, condition-stratified, grouped benchmark.

    Pairs are never split across train/dev/test: a question and both of its
    answers land in the same split. Without this the task collapses into
    memorising which answer belongs to a seen question.
    """
    rng = random.Random(SEED)

    pairs: dict[str, list[dict]] = collections.defaultdict(list)
    for r in pool:
        if r["pair_complete"]:
            pairs[r["pair_id"]].append(r)

    by_condition: dict[str, list[str]] = collections.defaultdict(list)
    for pid, members in pairs.items():
        by_condition[members[0]["condition"]].append(pid)

    target_pairs = TARGET_RECORDS // 2
    wanted = {
        "has_context": round(target_pairs * HAS_CONTEXT_SHARE),
        "no_context": target_pairs - round(target_pairs * HAS_CONTEXT_SHARE),
    }

    selected: list[str] = []
    for condition, n_wanted in wanted.items():
        candidates = sorted(by_condition[condition])
        rng.shuffle(candidates)
        cap = int(n_wanted * MAX_SUBJECT_SHARE)
        used = collections.Counter()
        chosen = []
        # First pass respects the per-subject cap; the second backfills if the
        # cap left us short, so the split sizes stay exact.
        for pid in candidates:
            subject = pairs[pid][0]["subject"]
            if len(chosen) >= n_wanted:
                break
            if used[subject] >= cap:
                continue
            used[subject] += 1
            chosen.append(pid)
        if len(chosen) < n_wanted:
            remaining = [p for p in candidates if p not in set(chosen)]
            chosen.extend(remaining[: n_wanted - len(chosen)])
        if len(chosen) < n_wanted:
            raise ValueError(
                f"pool exhausted for {condition}: wanted {n_wanted}, got {len(chosen)}"
            )
        selected.extend(chosen)

    # Stratify the train/dev/test assignment by (condition, subject) so every
    # split carries the same composition.
    strata: dict[tuple, list[str]] = collections.defaultdict(list)
    for pid in selected:
        m = pairs[pid][0]
        strata[(m["condition"], m["subject"])].append(pid)

    splits: dict[str, list[str]] = {"train": [], "dev": [], "test": []}
    order = [("dev", SPLIT_SIZES["dev"]), ("test", SPLIT_SIZES["test"])]
    total_pairs = len(selected)
    for stratum in sorted(strata):
        members = sorted(strata[stratum])
        rng.shuffle(members)
        cursor = 0
        for name, size in order:
            take = round(len(members) * (size // 2) / total_pairs)
            splits[name].extend(members[cursor : cursor + take])
            cursor += take
        splits["train"].extend(members[cursor:])

    return {
        name: [r for pid in sorted(pids) for r in sorted(pairs[pid], key=lambda x: x["id"])]
        for name, pids in splits.items()
    }


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def describe(name: str, rows: list[dict]) -> str:
    labels = collections.Counter(r["label"] for r in rows)
    conditions = collections.Counter(r["condition"] for r in rows)
    n = max(len(rows), 1)
    return (
        f"{name:<8} n={len(rows):<6} "
        f"hallucinated={labels[0] / n:.1%} "
        f"has_context={conditions['has_context'] / n:.1%} "
        f"pairs={len({r['pair_id'] for r in rows})}"
    )


def main() -> None:
    raw = load_raw()
    cleaned, stats = clean(raw)
    pool = to_schema(cleaned)
    write_jsonl(INTERIM, pool)

    print("cleaning:", json.dumps(stats, indent=2))
    print(f"\nwrote {INTERIM} ({len(pool):,} records)")

    splits = build_benchmark(pool)
    for name, rows in splits.items():
        write_jsonl(SPLIT_DIR / f"{name}.jsonl", rows)
        print("  " + describe(name, rows))

    # Leakage guard: a pair must never span two splits.
    seen: dict[str, str] = {}
    for name, rows in splits.items():
        for r in rows:
            other = seen.setdefault(r["pair_id"], name)
            if other != name:
                raise AssertionError(f"pair {r['pair_id']} spans {other} and {name}")
    print("\nleakage check passed: no pair_id spans two splits")


if __name__ == "__main__":
    main()
