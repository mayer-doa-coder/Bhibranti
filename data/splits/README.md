# `data/splits/` — the locked Phase 1 splits

Built by [`src/build_corpus.py`](../../src/build_corpus.py) from `data/corpus/bn_v1/corpus.jsonl`.
Deterministic, seed 42.

| Split | pairs | records |
|---|---:|---:|
| `train.jsonl` | 3,129 | 8,044 |
| `dev.jsonl` | 673 | 1,732 |
| `test.jsonl` | 678 | 1,740 |

Each split is 60% has-context and 50/50 correct vs hallucinated.

## Two rules

**1. Splits are grouped by pair.** A question and both of its answers always land in the same
split. Splitting by record would put the correct answer in train and the hallucinated one in
test, so a model could score by remembering the question instead of judging the answer.

**2. `test.jsonl` is touched exactly once**, at M6, for the final numbers. Use `dev.jsonl` as
often as you like. Every look at the test set that changes a decision spends part of it.

## `_pool_sanity/`

Throwaway splits from `src/build_bn_pool.py`, used only to check that script still runs. Not a
deliverable, git-ignored. Do not train on them or report anything from them.
