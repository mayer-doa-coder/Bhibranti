"""Turn a record into the exact text a model reads: the three input formats (guide §6, M9).

Which of the three works best is one of the questions this project answers, so every
model gets its input from here rather than building its own string.

    from preprocess import as_tokens, format_record, record_parts

    as_tokens(record, "F2")        -> ['পাঠ্য', ..., '<SEP>', 'প্রশ্ন', ..., '<SEP>', 'উত্তর']
    format_record(record, "F3")    -> ModelInput(text_a="<passage>", text_b="<question> <answer>")
    record_parts(record)           -> Parts(context=..., question=..., answer=...)

Run from the repository root:

    python src/preprocess.py --demo                       # all three formats on one real record
    python src/preprocess.py --check                      # the guarantees below, on train + dev
    python src/preprocess.py --format F3 --split dev --out data/processed/

THE THREE FORMATS
    F1  question + answer                     the only choice when there is no passage
    F2  passage + question + answer           everything in one piece of text
    F3  passage  ||  question + answer        two segments, the way an entailment model
                                              expects: "does this passage support this?"

THE ONE RULE THAT MATTERS: CUT THE PASSAGE, NEVER THE QUESTION OR THE ANSWER
A model that never sees the answer cannot judge it, and a truncated question changes what
was asked. So the question and answer are reserved first and the passage gets what is left.

WHY THE PASSAGE IS CUT FROM THE END, AND NOT SOMETHING CLEVERER
Measured on train + dev, using the pairs whose answer is a literal span of their passage
(so the evidence can be located):

    budget 256 tokens : 7 of 1,634 pairs lose their evidence   (0.4%)
    budget 128 tokens : 117 of 1,634 lose it                   (7.2%)

Keeping the window that best matches the QUESTION instead rescues 2 of those 7. Not worth
the extra machinery. Picking the window that best matches the ANSWER would rescue more and
is FORBIDDEN: it would quietly hand the model the evidence for correct answers only, which
is the string-matching shortcut all over again (PRD §5.5).

At the default budget of 256, only 1.5% of has-context records need cutting at all.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Callable, NamedTuple

from text_bn import TOKEN_PATTERN, clean, preprocess as tokenise_variant, tokenize

# =============================================================================
# SETTINGS
# =============================================================================

# Longest input a model reads, counted in tokens. 256 is the project default
# (CLAUDE.md, guide §7.1c and §7.3). Raising it costs memory; lowering it loses evidence.
MAX_TOKENS = 256

# Marker between the parts, for models that read one flat list of tokens.
# It is added as its own token so a tokenizer can never split it up.
SEPARATOR = "<SEP>"

# What each format contains. "pair" means the two segments stay separate (F3).
FORMATS = {
    "F1": "question + answer (no passage)",
    "F2": "passage + question + answer, all in one",
    "F3": "passage as one segment, question + answer as the other",
}

PROCESSED_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"


class Parts(NamedTuple):
    """A record's three pieces of text, with the passage already cut to fit."""
    context: str
    question: str
    answer: str


class ModelInput(NamedTuple):
    """What a model is given. text_b is None unless the format keeps two segments (F3)."""
    text_a: str
    text_b: str | None = None

    def flat(self) -> str:
        """Both segments as one string, for models that cannot take a pair."""
        return self.text_a if self.text_b is None else f"{self.text_a} {SEPARATOR} {self.text_b}"


# =============================================================================
# TRUNCATION - the part that can silently ruin a record
# =============================================================================

def count_tokens(text: str) -> int:
    """Default length measure: this project's own Bengali tokenizer."""
    return len(tokenize(text))


def cut_to_tokens(text: str, limit: int) -> str:
    """Keep the first `limit` tokens of text, cutting at a real character boundary.

    Rejoining tokens with spaces would NOT give back the original text - punctuation is
    its own token - so the cut is made using each token's own position in the string.

    >>> cut_to_tokens("এক দুই তিন চার", 2)
    'এক দুই'
    """
    if limit <= 0:
        return ""
    ends = [match.end() for match in TOKEN_PATTERN.finditer(text)]
    return text if limit >= len(ends) else text[:ends[limit - 1]].rstrip()


def record_parts(record: dict, budget: int = MAX_TOKENS, raw: bool = False,
                 length_fn: Callable[[str], int] | None = None) -> Parts:
    """A record's three pieces, with ONLY the passage shortened if the budget is tight.

    `budget` counts tokens. `length_fn` lets a caller measure with a different tokenizer
    (a transformer's sub-word one splits Bengali words into several pieces, so it counts
    more than ours does - pass it in to pre-cut exactly).

    `raw=True` skips the text cleaning, for the V0 variant of the M10 experiment.
    """
    tidy = (lambda t: t or "") if raw else clean
    measure = length_fn or count_tokens

    question, answer = tidy(record["question"]), tidy(record["candidate_answer"])
    context = tidy(record["context"])
    if not context:
        return Parts("", question, answer)

    # The question and the answer are reserved first; 2 separators are allowed for.
    reserved = measure(question) + measure(answer) + 2 * measure(SEPARATOR)
    room = budget - reserved
    if room >= measure(context):
        return Parts(context, question, answer)
    return Parts(cut_to_tokens(context, max(room, 0)), question, answer)


# =============================================================================
# THE THREE FORMATS
# =============================================================================

def format_record(record: dict, fmt: str = "F2", budget: int = MAX_TOKENS,
                  raw: bool = False, length_fn: Callable[[str], int] | None = None) -> ModelInput:
    """Build the model's input in one of the three formats.

    A record with no passage always comes out as F1, whichever format was asked for -
    there is simply nothing else to put in.
    """
    if fmt not in FORMATS:
        raise ValueError(f"unknown format {fmt!r}; choose from {list(FORMATS)}")

    parts = record_parts(record, budget, raw, length_fn)
    question_answer = f"{parts.question} {parts.answer}".strip()

    if fmt == "F1" or not parts.context:
        return ModelInput(question_answer)
    if fmt == "F2":
        return ModelInput(f"{parts.context} {question_answer}".strip())
    return ModelInput(parts.context, question_answer)      # F3 keeps the two apart


def as_text(record: dict, fmt: str = "F2", **kwargs) -> str:
    """The input as a single string (F3's two segments joined by the separator)."""
    return format_record(record, fmt, **kwargs).flat()


def as_tokens(record: dict, fmt: str = "F2", variant: str = "V1",
              budget: int = MAX_TOKENS, length_fn: Callable[[str], int] | None = None) -> list[str]:
    """The input as a token list, for the models that are not pretrained encoders.

    `variant` is the M10 preprocessing variant from text_bn (V1 is the default: clean
    and split only). The separator is kept as one token, never split up.
    """
    parts = record_parts(record, budget, raw=(variant == "V0"), length_fn=length_fn)
    pieces = [parts.question, parts.answer] if fmt == "F1" or not parts.context else (
        [parts.context, parts.question, parts.answer])

    tokens: list[str] = []
    for piece in pieces:
        if tokens:
            tokens.append(SEPARATOR)
        tokens.extend(tokenise_variant(piece, variant))
    return tokens


# =============================================================================
# WRITING THE FORMATTED SPLITS TO DISK
# =============================================================================

def dump_split(split: str, fmt: str, out_dir: Path, budget: int = MAX_TOKENS) -> Path:
    """Write one split in one format, for inspection or for a notebook to pick up.

    Only the fields a model or a results table needs are written. The label-revealing
    fields (`hallucination_type`, `annotator_*`, ...) are deliberately left out.
    """
    from splits import load_split

    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{split}.{fmt}.jsonl"
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        for record in load_split(split):
            model_input = format_record(record, fmt, budget)
            fh.write(json.dumps({
                "id": record["id"],
                "pair_id": record["pair_id"],
                "text_a": model_input.text_a,
                "text_b": model_input.text_b,
                "label": record["label"],
                "condition": record["condition"],
                "difficulty": record["difficulty"],
                "subject": record["subject"],
            }, ensure_ascii=False) + "\n")
    return path


# =============================================================================
# COMMAND LINE
# =============================================================================

def run_demo() -> int:
    """Show one has-context record in all three formats, and one no-context record."""
    from splits import load_split

    dev = load_split("dev")
    for condition in ("has_context", "no_context"):
        record = next(r for r in dev if r["condition"] == condition
                      and 40 < len(clean(r["context"] or r["question"])) < 400)
        print("=" * 78)
        print(f"{condition} / {record['subject']} / label = {record['label']} "
              f"({'correct' if record['label'] == 1 else 'hallucinated'})")
        if record["context"]:
            print(f"  passage : {clean(record['context'])[:100]}...")
        print(f"  question: {clean(record['question'])[:100]}")
        print(f"  answer  : {clean(record['candidate_answer'])[:60]}")
        print()
        for fmt in FORMATS:
            model_input = format_record(record, fmt)
            print(f"  {fmt} - {FORMATS[fmt]}")
            print(f"     text_a: {model_input.text_a[:95]}")
            if model_input.text_b is not None:
                print(f"     text_b: {model_input.text_b[:95]}")
            print(f"     tokens: {count_tokens(model_input.flat())}")
        print(f"\n  as_tokens(record, 'F2')[:12] = {as_tokens(record, 'F2')[:12]}")
    return 0


def run_check() -> int:
    """The guarantees, checked on every train and dev record."""
    from splits import load_split

    train, dev = load_split("train"), load_split("dev")
    records = train + dev
    failures: dict[str, int] = {}
    examples: dict[str, str] = {}

    def fail(rule: str, detail: str) -> None:
        failures[rule] = failures.get(rule, 0) + 1
        examples.setdefault(rule, detail)

    cut_counts = {fmt: 0 for fmt in FORMATS}
    for record in records:
        answer, question = clean(record["candidate_answer"]), clean(record["question"])
        for fmt in FORMATS:
            text = as_text(record, fmt)
            if answer not in text:
                fail("the answer is always in the input", f"{fmt} {record['id']}")
            if question not in text:
                fail("the question is always in the input", f"{fmt} {record['id']}")
            if record["context"]:
                kept = record_parts(record).context
                if kept and kept not in clean(record["context"]):
                    fail("the passage is only ever shortened, never altered", record["id"])
                if kept != clean(record["context"]):
                    cut_counts[fmt] += 1
        # tokens and text must describe the same thing
        if as_tokens(record, "F1")[-1] != tokenize(answer)[-1]:
            fail("the token list ends with the answer", record["id"])

    # The two records of a pair must differ ONLY by the answer. Comparing the finished
    # strings will not do: an answer often also appears inside the passage, so cutting it
    # out of both strings changes the passage too. Compare the parts instead.
    pairs: dict[str, list[dict]] = {}
    for record in records:
        pairs.setdefault(record["pair_id"], []).append(record)
    uneven_passage = 0
    for pair in pairs.values():
        if len(pair) != 2:
            continue
        first, second = (record_parts(r) for r in pair)
        if first.question != second.question:
            fail("a pair's two inputs share the same question", pair[0]["pair_id"])
        if first.context != second.context:
            # Allowed only when the budget forced a cut: each record reserves room for its
            # own answer, so a longer answer leaves less room for the passage. One must
            # then be a prefix of the other.
            longer, shorter = sorted((first.context, second.context), key=len, reverse=True)
            if not longer.startswith(shorter):
                fail("a pair's two inputs share the same passage", pair[0]["pair_id"])
            uneven_passage += 1

    print("=" * 78)
    print(f"PREPROCESS CHECK - {len(records):,} train+dev records, budget {MAX_TOKENS} tokens")
    print("=" * 78)

    grounded = [r for r in records if r["context"]]
    print(f"  has-context records whose passage had to be cut: {cut_counts['F2']:,} of "
          f"{len(grounded):,} ({cut_counts['F2'] / len(grounded):.1%})")

    lengths = {fmt: sorted(count_tokens(as_text(r, fmt)) for r in records) for fmt in FORMATS}
    print(f"\n  {'format':<6}{'median':>9}{'p90':>7}{'p99':>7}{'max':>7}   what it holds")
    for fmt, sizes in lengths.items():
        print(f"  {fmt:<6}{sizes[len(sizes) // 2]:>9}{sizes[int(len(sizes) * .9)]:>7}"
              f"{sizes[int(len(sizes) * .99)]:>7}{sizes[-1]:>7}   {FORMATS[fmt]}")
    over = sum(1 for n in lengths["F2"] if n > MAX_TOKENS)
    print(f"\n  inputs still longer than the budget after cutting: {over:,} "
          f"(these are records whose question + answer alone are that long - "
          f"cutting them is not allowed)")

    print(f"  pairs whose two records keep a different amount of passage: {uneven_passage:,} "
          f"(only possible when the budget forced a cut; each record reserves room for its "
          f"own answer)")

    print("\n  RULES")
    for rule in ["the answer is always in the input",
                 "the question is always in the input",
                 "the passage is only ever shortened, never altered",
                 "the token list ends with the answer",
                 "a pair's two inputs share the same question",
                 "a pair's two inputs share the same passage"]:
        count = failures.get(rule, 0)
        print(f"    [{'PASS' if not count else f'FAIL x{count}'}] {rule}")
        if rule in examples:
            print(f"           e.g. {examples[rule]}")
    print("=" * 78)
    return 1 if failures else 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the model input formats (guide §6, M9).")
    parser.add_argument("--demo", action="store_true", help="show all three formats on real records")
    parser.add_argument("--check", action="store_true", help="verify the guarantees on train + dev")
    parser.add_argument("--format", choices=list(FORMATS), help="format to write out")
    parser.add_argument("--split", choices=["train", "dev", "test"], help="split to write out")
    parser.add_argument("--out", type=Path, default=PROCESSED_DIR, help="where to write")
    parser.add_argument("--budget", type=int, default=MAX_TOKENS, help="token budget")
    args = parser.parse_args()

    if args.check:
        return run_check()
    if args.demo:
        return run_demo()
    if args.format and args.split:
        if args.split == "test":
            print("Refusing: test.jsonl is opened once, at the final evaluation (PRD §5.4).")
            return 1
        path = dump_split(args.split, args.format, args.out, args.budget)
        print(f"wrote {path}")
        return 0
    parser.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
