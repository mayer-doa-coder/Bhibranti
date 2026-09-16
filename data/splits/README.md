# `data/splits/` — the locked Phase 1 splits

Built by [`src/build_corpus.py`](../../src/build_corpus.py) from `data/corpus/bn_v1/corpus.jsonl`.
Deterministic, seed 42.

| Split | pairs | records |
|---|---:|---:|
| `train.jsonl` | 3,129 | 6,258 |
| `dev.jsonl` | 673 | 1,346 |
| `test.jsonl` | 678 | 1,356 |

70 / 15 / 15 by pair, split separately inside every (condition, subject) group, so each split is
60% has-context and 50/50 correct vs hallucinated, with the same subject mix.

**Excluded pairs (PRD §5.1d).** 180 pairs that human review found broken or wrongly labelled are
marked `excluded = true` and left out of training and scoring. Usable: train 3,067 pairs / 6,134
records · dev 619 / 1,238 · test 614 / 1,228. **Load data with `src/splits.py`** — never read these
files directly in a training or evaluation script.

**Annotation lives in these files after M3.** `src/merge_annotation.py` writes
`hallucination_type`, `type_source`, `annotator_1/2`, `adjudicated`, `excluded` and
`exclusion_reason` into them from
`data/annotated/round1/`. Rebuilding with `src/build_corpus.py` resets those fields, so rerun the
merge after any rebuild.

## Two rules

**1. Splits are grouped by pair.** A question and both of its answers always land in the same
split. Splitting by record would put the correct answer in train and the hallucinated one in
test, so a model could score by remembering the question instead of judging the answer.

**2. `test.jsonl` is touched exactly once**, at M6, for the final numbers. Use `dev.jsonl` as
often as you like. Every look at the test set that changes a decision spends part of it.

## `_pool_sanity/`

Throwaway splits from `src/build_bn_pool.py`, used only to check that script still runs. Not a
deliverable, git-ignored. Do not train on them or report anything from them.
