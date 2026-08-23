# `data/splits/`

## The real Phase 1 splits — NOT YET CREATED

`train.jsonl`, `dev.jsonl`, `test.jsonl` live here and are created at **milestone M3**, from
annotated Banglish data in `data/annotated/`, by a deterministic script (PRD R6).

| Split | Target | Annotation |
|---|---:|---|
| `train.jsonl` | ~3,000 | generator label + 20% human verification |
| `dev.jsonl` | ~500 | 100% human, single annotator |
| `test.jsonl` | ~500 | **100% human, double-annotated, adjudicated** |

Composition: 60/40 has-context/no-context, 50/50 class balance (±5%), ~60/40 easy/hard.
Labels follow PRD §5.1a: **`1` = correct/faithful, `0` = incorrect/hallucinated**.

### The test set lock

`test.jsonl` is **designated and locked at M0** and **opened exactly once, at M6** (PRD E9,
RK10). Every intermediate decision — model choice, arm, format, threshold — is made on `dev` or
by 5-fold CV on train+dev.

Repeated test evaluation is the quiet way this project's headline number becomes meaningless.
There is no partial credit for "just checking once more".

---

## `bn_dev_benchmark/` — pipeline development only

A working benchmark built from the Bengali-script source pool so the model ladder can be
developed **now**, in parallel with the corpus work.

| | records | pairs | hallucinated | has-context |
|---|---:|---:|---:|---:|
| `train.jsonl` | 3,112 | 1,496 | 51.1% | 57.9% |
| `dev.jsonl` | 531 | 252 | 51.4% | 57.8% |
| `test.jsonl` | 538 | 252 | 50.7% | 56.9% |

Built by `src/build_bn_pool.py` (seed 42, fully deterministic). Grouped on `pair_id` — verified
that no pair spans two splits. Subject-capped at 25% per condition, because mathematics is 52%
of the raw pool and would otherwise dominate.

> Record counts exceed 2 × pairs because some questions in the source pool carry more than two
> candidate answers. All members of a pair stay in the same split.

### This is NOT the Phase 1 corpus

- **Bengali script, not Banglish** — 0.96% Latin characters. The project's core premise cannot
  be tested on it.
- **Not human-annotated** — no `hallucination_type`, no `difficulty`, no κ.
- **Carries a severe shortcut** — a substring rule scores **0.832 macro-F1** on the has-context
  half, above the project's 0.80 target, with no learning at all.

**Never promote these files into the parent directory.** Any number reported from them must be
labelled Bengali-script pipeline validation, not a Phase 1 result.

### What it is good for

Milestone **M4** — get `train_classical.py`, `train_transformer.py`, `evaluate.py`, the 5-fold
CV harness, the 3-seed loop, and `results/experiment_log.csv` working end-to-end on real data.
It also gives a Bengali-script reference point that becomes a genuinely useful comparison later:
the same detector on the same questions, Bengali script vs Banglish.

```bash
python src/audit.py --data data/splits/bn_dev_benchmark
```
