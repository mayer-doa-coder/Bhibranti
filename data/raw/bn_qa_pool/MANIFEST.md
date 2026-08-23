# Raw pool manifest — `bn_qa_pool`

Source files, byte-identical to what was supplied in the original `small dataset/`
folder. **Never edit these.** Cleaning and schema mapping happen downstream in
`data/interim/bn_pool.jsonl`, produced by `src/build_bn_pool.py`.

Content origin: **Bengali Wikipedia** (CC BY-SA 4.0 — attribution + share-alike) and
**BCS question banks**. QA pairs were built from those texts with LLM assistance.
Full register and licence constraints: [`data/SOURCES.md`](../../SOURCES.md).

Source schema: `{context, prompt_bn, response_bn, label}`.

**Label polarity here is `1 = correct`, `0 = hallucinated` — the same convention the
project uses (PRD section 5.1a). No flip is applied anywhere in the pipeline.**

Do not load these files directly into a model: they contain 8.9% exact duplicates,
68 contradictory records, and none of the schema fields. Use `data/interim/`.

| File | Records | has-context | no-context | correct (label=1) | SHA-256 (first 16) |
|---|---:|---:|---:|---:|---|
| `small_antonym_training_data.jsonl` | 2,000 | 27 | 1,973 | 1,000 | `84c607520145cda0` |
| `small_bcs_training_data.jsonl` | 2,324 | 0 | 2,324 | 1,162 | `3b902c62c5bde91d` |
| `small_geography_training_data.jsonl` | 1,741 | 1,643 | 98 | 899 | `f61e25d109e4e20a` |
| `small_grammar_training_data.jsonl` | 5,000 | 712 | 4,288 | 2,500 | `e4bd16ea01d5b18a` |
| `small_history_training_data.jsonl` | 5,000 | 3,377 | 1,623 | 2,500 | `11bea1a5a6475b86` |
| `small_idiom_meaning_training_data.jsonl` | 2,000 | 0 | 2,000 | 1,000 | `85afbfd12a072f8d` |
| `small_law_training_data.jsonl` | 1,383 | 1,117 | 266 | 824 | `dde3a51e276a6417` |
| `small_literature_training_data.jsonl` | 914 | 75 | 839 | 479 | `2083403440d31051` |
| `small_mathematics_training_data.jsonl` | 29,332 | 0 | 29,332 | 14,666 | `283b7fde1805d3d9` |
| `small_others_training_data.jsonl` | 390 | 340 | 50 | 200 | `20602e2c3d1b821e` |
| `small_reading_comprehension_training_data.jsonl` | 5,000 | 5,000 | 0 | 2,500 | `5b2f9c7246ba5490` |
| `small_science_training_data.jsonl` | 4,000 | 57 | 3,943 | 2,000 | `c3ce28c0ec4d555a` |
| `small_synonym_training_data.jsonl` | 1,000 | 0 | 1,000 | 500 | `ec44fe7ee67794ab` |
| `small_vocabulary_training_data.jsonl` | 2,000 | 0 | 2,000 | 1,000 | `63134672411e0b01` |
| **Total** | **62,084** | **12,348** | **49,736** | **31,230** | |

Verify with:

```bash
sha256sum data/raw/bn_qa_pool/*.jsonl
```
