# `data/interim/` — cleaned, schema-mapped pool

**`bn_pool.jsonl` — 56,480 records.** Produced by `src/build_bn_pool.py` from
`data/raw/bn_qa_pool/`. Regenerate at any time; never edit by hand.

This stage exists because the raw pool needs cleaning before it is usable, and that cleaning must
be visible and reproducible rather than baked into a one-off notebook cell.

| Correction | Detail |
|---|---|
| Exact duplicates dropped | 5,536 records |
| Contradictory records dropped | 68 records (identical text carrying both labels) |
| Mapped to the frozen schema | ids, `pair_id`, condition, provenance, CMI proxy |

**Labels are NOT changed.** The project convention is `1 = correct/faithful`, `0 = incorrect/
hallucinated` (PRD §5.1a), which is exactly what the source files use. Nothing is flipped
anywhere in the pipeline.

Records are grouped into `pair_id` — one question with its faithful and hallucinated answers.
`pair_complete` is true when both labels are present (19,420 groups). **Splits must be grouped
on `pair_id`**; splitting a pair across train and test turns the task into memorisation.

## What this is not

This is **Bengali script, not Banglish** — 0.96% Latin characters. It is the base QA layer that
the Banglish condition gets produced *from* (guide §3.3), not the Phase 1 corpus. Fields reflect
that honestly:

- `script_condition: "bengali_script"`
- `cmi` computed by a **script proxy** and ≈ 0 throughout; recompute with real token-level
  language ID once Banglish text exists
- `hallucination_type: "unlabeled"` on positives — provisional, never ship it
- `difficulty: "unlabeled"`, `annotator_1/2: null`

See `data/DATASET_AUDIT.md` for the full analysis.
