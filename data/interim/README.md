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

This is **Bengali script** — 0.96% Latin characters — which is exactly what Phase 1 needs.

It is the **whole** pool, not the corpus. `src/build_corpus.py` pairs it, drops only invalid
records, and applies the 60/40 and per-subject composition constraints to produce
`data/corpus/bn_v1/corpus.jsonl` (4,480 pairs). Use that for anything that trains or
evaluates. Fields reflect
that honestly:

- `script_condition: "bengali_script"`
- `cmi` computed by a **script proxy** and ≈ 0 throughout; recompute with real token-level
  language ID in Phase 2, once Banglish text exists
- `hallucination_type: "unlabeled"` on positives — provisional, never ship it
- `difficulty: "unlabeled"`, `annotator_1/2: null`

See `data/DATASET_AUDIT.md` for the full analysis.
