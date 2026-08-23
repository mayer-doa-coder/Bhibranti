# `data/generated/` — Banglish candidates

## `pilot_v1/` — 500-pair draft, ready for human review

Built. See **[`pilot_v1/README.md`](pilot_v1/README.md)** for everything: how the 500 pairs
were picked, how the Banglish draft was produced, what it's good/bad at, and exactly how to
review it. Open **`pilot_v1/review_sheet.csv`** to start reviewing.

The M1 gate already passes on this draft (metadata probe 0.505 < 0.60). What's left is your
manual read-through (Task 3) before this counts as real data.

---

**Below: the original plan for scaling past the pilot, once M1 is confirmed closed.**

Populated by `src/generate.py` at milestone **M1**, from the base QA layer in
`data/interim/bn_pool.jsonl`. The step that has to happen here is guide §3.3 + §4: turn
Bengali-script QA pairs into **naturalistic Bangla–English code-mixed** ones, and produce
faithful/hallucinated answer pairs under the controlled protocol.

## Before generating at scale

1. **Fix the copy-from-context shortcut.** The source pool's faithful answers appear verbatim in
   the context 78.6% of the time, which lets a string matcher score 0.832 macro-F1. Do not
   inherit it. Ask the prompt for faithful answers that *paraphrase* the context, and for
   hallucinated answers that reuse its surface tokens (the `relational` and `contradiction`
   types do this naturally). See `data/DATASET_AUDIT.md` Finding 4.
2. **Generate both classes in one call per item**, same template, same temperature, same length,
   same register — only factual content differs (guide §4.1).
3. **Enforce length parity programmatically** (`tol=0.15`) and log the rejection rate.
4. **Capture `error_span`** — one extra JSON key now, span-level supervision for Phase 2.
5. **Version-control the prompt** (PRD R3). A prompt change invalidates everything generated
   under the old one.

## Then, before anything downstream

```bash
python src/audit.py --data data/generated/pilot_v1/pilot.jsonl --probe metadata
```

**M1 is a hard gate: the pilot's metadata probe must score < 0.60 macro-F1.** No progression to
M3 without it. If it fails, inspect the coefficients, fix the prompt, regenerate — never move
the threshold.

## Mixing routes (target ≥ 500 human-written items, PRD D6)

| Route | Notes |
|---|---|
| Native writers | Highest quality, most naturalistic — this is what separates you from synthetic benchmarks |
| Transliteration tooling | `bnbphoneticparser`, `indic_transliteration`, `bntranslit`; spot-check ~10% |
| LLM rewriting | Instruct it to preserve natural spelling variance |

Automated transliteration produces unnaturally *consistent* spelling. Real Banglish is
inconsistent, and that inconsistency is the actual difficulty being measured.
