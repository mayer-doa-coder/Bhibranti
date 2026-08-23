# `data/corpus/` — the filtered Phase 1 corpus

## `bn_v1/corpus.jsonl` — 4,480 pairs / 8,960 records

The Bengali corpus, built from `data/interim/bn_pool.jsonl` by
[`src/build_corpus.py`](../../src/build_corpus.py). Read that script's docstring before changing
anything here — it explains every filter and why it exists.

**Nothing is excluded for being hard.** Law, science, BCS and literature are all here on equal
footing — 965 pairs, 21.5% of the corpus.

**This is QA only: fill-in-the-blank (cloze) items are excluded.** That is a task-type rule, not
a difficulty one — cloze teaches span-copying rather than answer checking. It removes `geography`
entirely, since every geography item was fill-in-the-blank. 13 subjects remain.

Beyond that, only four things are removed: incomplete pairs, OCR-damaged text, questions copied
verbatim out of their own passage, and duplicate question text.

| | pairs | |
|---|---:|---|
| has-context | 2,688 | 60% — answer must be supported by the passage |
| no-context | 1,792 | 40% — language, arithmetic, and general knowledge |
| **total** | **4,480** | 50/50 correct vs hallucinated |

`difficulty` marks the shortcut-proof pairs: **hard** means "does the answer appear in the
passage?" does not separate the correct answer from the wrong one, so a model has to actually
read. 927 has-context pairs are hard; 407 no-context pairs are hard (near-miss wrong answers).

**Not yet annotated.** Every record carries `hallucination_type='unlabeled'`. The type field
becomes real only after M2 passes and people label it.

Splits are built from this file into `data/splits/` in the same run, grouped so a pair never
straddles two splits.
