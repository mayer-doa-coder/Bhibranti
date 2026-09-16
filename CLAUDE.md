# CLAUDE.md — Phase 1 Cheat Sheet

**Phase 1 is Bengali only.** Bangla script, Bangla questions, Bangla answers. The job is to
detect when an answer is wrong (hallucinated).

**Banglish comes later, in Phase 2.** Do not start it. Full context in [AGENT.md](AGENT.md);
requirements in [docs/PRD.md](docs/PRD.md); recipes in
[docs/IMPLEMENTATION_GUIDE.md](docs/IMPLEMENTATION_GUIDE.md).

---

## Hard constraints

| | |
|---|---|
| **RAG is forbidden.** | No retrieval-augmented anything. No vector store, no nearest-neighbour lookup over the corpus, no retrieving passages or similar examples at inference. The model sees the question, the passage that is already in the record, and the answer. Nothing else. |
| **Allowed** | N-gram · Skip-gram · word2vec · word & text embeddings · RNN · LSTM/BiLSTM (+attention) · BERT-family encoders · anything else that is not RAG. Adding techniques is fine; the required ladder below may not be *replaced*. |
| **Data collection is finished.** | The corpus is final. Do not collect, scrape, generate, or synthesise more data. Work with `data/corpus/bn_v1/`. |
| **QA only — no fill-in-the-blank.** | Cloze / শূন্যস্থান পূরণ items are excluded. They teach span-copying rather than answer checking. This removed `geography` entirely, which was 100% cloze. |

---

## Current status

**M0 done. M1 done. M2 done (kappa = 0.717). M3 done (test kappa = 0.865, merged, D8 met,
180 flagged pairs excluded). Next up: M4 — the model ladder.**

**Usable data (load it only through `src/splits.py`):** train 3,067 pairs / 6,134 records ·
dev 619 / 1,238 · test 614 / 1,228.

- [x] Repo scaffolded, `configs/schema.json` frozen, seeds fixed at **42, 1337, 2024**
- [x] 62,084 raw records → cleaned to 56,480 in `data/interim/bn_pool.jsonl`
- [x] **Corpus built → `data/corpus/bn_v1/corpus.jsonl`, 4,480 pairs / 8,960 records**
- [x] Splits built, grouped by pair → `data/splits/{train,dev,test}.jsonl` (3,129 / 673 / 678 pairs)
- [x] **Shortcut gate PASSES** — metadata probe 0.534 (needs < 0.60)
- [x] Easy/hard marked — the hard subset is where the string shortcut stops working
- [x] **M2 PASSED — Cohen's κ = 0.717** on 100 blind items, 2 annotators
      → [`data/annotated/agreement_test_v1/IAA_REPORT.md`](data/annotated/agreement_test_v1/IAA_REPORT.md)
- [x] Guidelines extended with Rules 11–15, taken from the actual disagreements
- [x] M3 sheets built → `data/annotated/round1/`
- [x] **All four annotation sets complete and validated**: test/tawhid (1,356), test/shejan
      (1,356), dev (1,346), train_spotcheck (1,252) — 5,310 rows, 0 structural errors, 0 tampered
      question/answer text, 0 cross-split leakage. Files use a `*_FINAL.csv` naming convention.
- [x] **Test split double-annotated — Cohen's κ = 0.865** (full 1,356 items, not a sample)
      → run `python src/score_test_agreement.py --a-dir .../test/tawhid --b-dir .../test/shejan`
- [x] **Merge tool rebuilt** — `src/merge_annotation.py` now reads the `*_FINAL.csv` sheets.
      Dry run verified; write path tested on a copy (labels/difficulty untouched, idempotent,
      gate still 0.534). It never changes `label` or `difficulty`.
- [x] **Adjudication done** — all 253 rows decided (141+2 typed, 89 dispute, 21 skip; 3 typo-pairs
      moved to skip on review). **Merge written 2026-09-17**; strict audit passes, gate 0.534.
      D8 MET: 1,867 of 1,977 in-scope wrong answers typed, 110 excluded by the adjudicator.
- [x] **D8 narrowed (PRD §5.1c):** types are required for test + dev + the 20% train spot-check
      (1,977 wrong answers), not the other 2,503 train records. Those stay `unlabeled` with
      `type_source = outside_train_sample`. Train *labels*: 4.5% confirmed noise (< 5% rule).
- [x] **PRD Q5 decided (§5.1d): human-flagged pairs are excluded from training AND scoring** —
      train 62, dev 54, test 64 (stored label confirmed wrong, or broken item). They stay in the
      files marked `excluded = true`; `src/splits.py` drops them. Gate on usable data: 0.514.
- [x] **Label disputes recorded, not applied:** 222 records where a human disagrees with the
      corpus label (116 unanimous) → `data/annotated/round1/label_disputes.csv`.
- [ ] Model ladder not built yet

**All 13 remaining subjects are in.** Nothing is excluded for being hard or for needing outside
knowledge — law, science, BCS and literature are all included on equal footing (966 pairs,
21.6% of the corpus). Difficulty is *measured* (see `difficulty`), not filtered away.

**`geography` is gone**, and that is not a difficulty judgement: every geography item was
fill-in-the-blank, and this project is QA only.

**Three numbers your has-context score must beat.** Report them, always. Measured on the usable
data (excluded pairs removed), because that is what models train and are scored on:

| Baseline | Score | What it means |
|---|---|---|
| String matcher, all has-context | **0.823** | "Does the answer appear in the passage?" — no learning at all |
| String matcher, easy subset | 0.987 | where the shortcut fully works |
| String matcher, **hard subset** | **0.454** | on hard pairs the shortcut is useless |

(Before exclusion they were 0.812 / 0.980 / 0.456. For a dev or test score, recompute the baseline
on that same split.) A has-context macro-F1 of 0.83 is **not** a result — it barely beats a string
matcher. The real
target is the **hard subset**, where 0.80 actually means something.

---

## Setup

```bash
python -m venv .venv && source .venv/Scripts/activate     # Windows/Git Bash
pip install -r requirements.txt        # Tier 1 only; Tier 2 is commented out until M4
```

## Commands

```bash
# Data pipeline — THESE FOUR EXIST AND WORK TODAY (deterministic, seed 42)
python src/build_bn_pool.py         # raw -> data/interim/bn_pool.jsonl
python src/build_corpus.py          # interim -> data/corpus/bn_v1/ + data/splits/
python src/build_agreement_test.py  # corpus -> 100 blind items for the M2 check
python src/audit.py --data data/splits              # all probes + structural checks

# THE GATE — metadata probe must be < 0.60, exits non-zero if not
python src/audit.py --data data/splits --probe metadata

# M3 — build the annotation sheets, then check filled ones
python src/build_annotation_sheets.py        # refuses to clobber labelled work
python src/validate_annotation.py --dir data/annotated/round1/test/tawhid
python src/score_test_agreement.py --a-dir data/annotated/round1/test/tawhid \
  --b-dir data/annotated/round1/test/shejan --a-name Tawhid --b-name Shejan   # kappa 0.865
python src/merge_annotation.py --dry-run     # sheets -> corpus report; drop --dry-run to write
                                             # also sets `excluded` on human-flagged pairs
                                             # rerun it after ANY build_corpus.py rebuild

# M2 gate — PASSED at kappa 0.717. Rerun any time:
python src/score_agreement.py \n  --a data/annotated/agreement_test_v1/items_for_annotation_tawhid.csv \n  --b data/annotated/agreement_test_v1/items_for_annotation_shejan.csv

# Not yet written
python src/preprocess.py --format F3 --in data/splits/ --out data/processed/
python src/train_classical.py    --model ngram_logreg --seed 42
python src/train_transformer.py  --model csebuetnlp/banglabert --seed 42
python src/evaluate.py --checkpoint out/best --split dev      # any number of times
python src/evaluate.py --checkpoint out/best --split test     # EXACTLY ONCE, at M6
```

> Every script takes `--seed` and appends a row to `results/experiment_log.csv`.

## Repo layout

| Path | Purpose |
|---|---|
| [docs/PRD.md](docs/PRD.md) | Requirements, gates, scope — **the authority** |
| [docs/IMPLEMENTATION_GUIDE.md](docs/IMPLEMENTATION_GUIDE.md) | Recipes, hyperparameters, code |
| [docs/ANNOTATION_GUIDELINES.md](docs/ANNOTATION_GUIDELINES.md) | Labelling rules — read before annotating |
| `configs/schema.json` | Frozen record schema |
| `src/build_bn_pool.py` | ✅ Cleans + dedups the raw pool |
| `src/build_corpus.py` | ✅ **Pairs, filters, splits.** Read its docstring first |
| `src/build_agreement_test.py` | ✅ Builds the 100-item blind test |
| `src/score_agreement.py` | ✅ Cohen's kappa — the M2 gate |
| `src/build_annotation_sheets.py` | ✅ Builds the M3 sheets. Read its docstring — 6 documented guards |
| `src/validate_annotation.py` | ✅ Checks filled sheets for the mistakes that actually happen |
| `src/score_test_agreement.py` | ✅ Cohen's kappa on the full test split |
| `src/merge_annotation.py` | ✅ Human sheets → `hallucination_type` + `excluded` in corpus + splits. Never touches `label` |
| `src/splits.py` | ✅ `load_split("train")` — **the only way to load data for training or scoring**; drops excluded pairs |
| [docs/PROJECT_WALKTHROUGH.md](docs/PROJECT_WALKTHROUGH.md) | Beginner-friendly explanation of every step so far — **keep it updated as steps finish** |
| `src/audit.py` | ✅ Shortcut probes + structural checks |
| `src/preprocess.py` | ⬜ Input formats F1/F2/F3 |
| `src/train_classical.py` | ⬜ N-gram, Skip-gram/word2vec, BiRNN, BiLSTM(+attn) |
| `src/train_transformer.py` | ⬜ Encoder fine-tuning |
| `src/further_pretrain.py` | ⬜ MLM further pretraining (mBERT / XLM-R only) |
| `src/evaluate.py` | ⬜ Metrics, breakdowns, McNemar, bootstrap CI |
| `data/DATASET_AUDIT.md` | **Read first.** Audit of the source pool |
| `data/corpus/bn_v1/corpus.jsonl` | ✅ The Phase 1 corpus, 4,480 pairs |
| `data/splits/` | ✅ `train/dev/test.jsonl` — the real splits |
| `data/annotated/round1/` | ✅ The M3 annotation sheets. Never open `_mapping_DO_NOT_SHARE.csv` |
| `data/interim/bn_pool.jsonl` | 56,480 cleaned records (the pool the corpus is drawn from) |
| `data/raw/bn_qa_pool/` | 14 source `.jsonl`, verbatim. Has duplicates — don't load directly |
| `data/SOURCES.md` | Licence register — 2 open questions |
| `results/experiment_log.csv` | Single source of truth for every run |

## Key facts

- `label`: **1 = correct/faithful**, **0 = hallucinated**. Positive class is *faithful* — the
  reverse of most papers. `probs[:, 1]` = P(correct).
- Corpus: 60% has-context / 40% no-context · 50/50 class balance · one pair = one correct + one
  hallucinated answer to the same question
- **Pairs never split across train/test.** Splitting by record would put the correct answer in
  train and the wrong one in test, so a model could score by memorising the question.
- `difficulty`: **hard** = the string shortcut does not separate the two answers on this pair.
  For no-context, hard = the wrong answer is a near-miss.
- **No fill-in-the-blank items.** QA only. They were 0.929 for a plain string matcher, which
  inflated every has-context score.
- Seeds: **42, 1337, 2024** — always all three, report mean ± std
- Fine-tune defaults: `lr=2e-5`, AdamW, `max_length=256`, `epochs=5`, `batch=32`, `warmup=0.1`,
  `weight_decay=0.01`, `fp16=True`, early stop on val loss (patience 2)
- Targets: has-context **≥ 0.80 on the hard subset**, no-context **≥ 0.60**, metadata probe
  **< 0.60**, κ **≥ 0.60**
- Taxonomy: intrinsic `entity|numeric|relational|contradiction`; extrinsic `fabricated|overclaim`;
  correct `none`. Types exist for test, dev and the train spot-check only (PRD §5.1c) — per-type
  results are reported on **dev/test only**
- **`hallucination_type`, `type_source`, `annotator_1/2`, `adjudicated` reveal the label — never a
  model input**

---

## DO

- **Report has-context and no-context separately**, and **easy and hard separately**. A single
  averaged number hides everything that matters here.
- **Load data with `from splits import load_split`.** It drops the 180 excluded pairs (PRD §5.1d).
  Reading `data/splits/*.jsonl` directly would train and score on labels humans confirmed wrong.
- **Quote the string baselines** (0.823 all / 0.454 hard on usable data, or recomputed on the
  split being scored) next to every has-context score you report.
- **Log every run** to `results/experiment_log.csv` — including crashes and failures.
- **Set and log a seed** in anything that touches randomness.
- **Run `normalize()`** from `csebuetnlp/normalizer` before BanglaBERT — it was pretrained with
  it, and skipping it costs points.
- **Break results down by subject too.** Law, science and BCS items are harder to verify; if the
  model collapses on those, that is a finding worth reporting, not something to hide.
- **Use McNemar + bootstrap CI** before claiming one model beats another.
- **Investigate any macro-F1 > 0.95** as leakage.

## DON'T

- **DON'T use RAG.** No retrieval at inference, no vector database, no nearest-neighbour lookup
  over training examples. The passage in the record is the only context the model gets.
- **DON'T collect or generate more data.** The corpus is final.
- **DON'T add fill-in-the-blank items back.** QA only — cloze is a different task.
- **DON'T touch `data/splits/test.jsonl`** except for the single sanctioned M6 evaluation.
- **DON'T weaken the shortcut probe** to make it pass — not the threshold, not the features, not
  the split. Fix the data and rebuild.
- **DON'T report a raw has-context number on its own.** A string matcher gets 0.823. Without the
  hard-subset number next to it, the score is meaningless.
- **DON'T use `AutoModelForMaskedLM` on BanglaBERT** — it is an ELECTRA discriminator. Further
  pretraining targets mBERT and XLM-R only.
- **DON'T swap out a required model.** The ladder (N-gram → Skip-gram → BiRNN/BiLSTM →
  BERT-family) is fixed by the course requirement. Adding is fine; replacing is not.
- **DON'T rename, remove, or repurpose a schema field.** `cmi`, `script_condition`, and
  `error_span` are Phase 2 payload: populate them, never evaluate on them.
- **DON'T hand-curate a split** or re-roll a seed until the numbers improve.
- **DON'T "fix" expected results.** Classical models at 0.50–0.65, and no-context well below
  has-context, are correct outcomes, not bugs.
- **DON'T start Phase 2 work** — Banglish/transliteration, CMI degradation, span-level
  evaluation, tokenizer fertility, paper writing. Flag it and stop.
- **DON'T commit API keys**, PII, or non-redistributable source data. Wikipedia content is
  CC BY-SA 4.0 — **share-alike**. Never mix it into one file with CC BY-**NC**-SA sources.
- **DON'T load `data/raw/bn_qa_pool/` directly** — 8.9% duplicates, no schema. Use `data/interim/`.
- **DON'T flip labels anywhere.** `1 = correct` end to end, source files included.

---

## When stuck

| Symptom | Check |
|---|---|
| Metadata probe > 0.60 | Inspect `clf.coef_` → find the leaking feature → fix the data → rebuild |
| has-context macro-F1 ≈ 0.81 | That is the string-matcher baseline. Look at the hard-subset score instead |
| macro-F1 > 0.95 | Leakage. Check for duplicate questions across splits; run the answer-only probe |
| Answer-only probe ≈ full model | Model isn't checking grounding — the answer text gives the label away |
| Hard subset far below easy subset | Expected and correct. That gap is the real finding |
| Model collapses on law / science / bcs | Expected — those need outside knowledge a closed-book model does not have. Report per-subject |
| κ < 0.60 | Guidelines, not annotators. Write explicit rules for the cases they disagreed on |
| Many `unsure` labels from one subject | A finding about the corpus. Report it; don't force a label |
| Model spread within 1–2 points | Expected. Use 3 seeds + McNemar; report ties honestly |
| CUDA OOM | `batch_size=16` + gradient accumulation. Don't change `max_length` mid-experiment |
| Score plateaued | ROI order: input format sweep → FPT → hard-negative balance → ensemble → threshold |
| Per-class metrics look swapped | `1` is the **correct** class here, not the hallucinated one |
