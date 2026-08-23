# CLAUDE.md — BanglishHallu Phase 1 Cheat Sheet

Bangla–English code-mixed ("Banglish") hallucination detection. Phase 1 = build the corpus + the
model ladder. Full context in [AGENT.md](AGENT.md); requirements in [docs/PRD.md](docs/PRD.md);
step-by-step recipes in [docs/IMPLEMENTATION_GUIDE.md](docs/IMPLEMENTATION_GUIDE.md).

---

## Current status

**Milestone M0 — Foundations, mostly done. A Bengali-script QA pool is ingested and audited.**

- [x] Repo tree scaffolded (`data/`, `src/`, `configs/`, `results/`, `notebooks/`)
- [x] `configs/schema.json` frozen
- [x] `results/experiment_log.csv` created, seeded with the audit runs
- [x] Seeds fixed: **42, 1337, 2024**
- [x] 62,084-record Bengali-script QA pool ingested → `data/raw/bn_qa_pool/`, cleaned to
      56,480 records in `data/interim/bn_pool.jsonl`
- [x] Dataset audited — **`data/DATASET_AUDIT.md` is required reading before touching the data**
- [x] Sources identified: Bengali Wikipedia (**CC BY-SA 4.0**) + BCS question banks
- [ ] **Two licence questions open:** BCS compilation terms, and the released-corpus licence
      (Wikipedia share-alike likely forces CC BY-SA 4.0). See `data/SOURCES.md`.
- [x] 500-pair Banglish pilot drafted → `data/generated/pilot_v1/`, **M1 metadata gate PASSES (0.505)**
- [ ] **Pilot needs human review** — `data/generated/pilot_v1/review_sheet.csv` (Task 3, not yet done)
- [ ] Generation prompt template with length + register constraints
- [x] `docs/ANNOTATION_GUIDELINES.md` written
- [x] 100-item agreement test built → `data/annotated/agreement_test_v1/`, waiting on 2 annotators
- [ ] Real Phase 1 test set designated and locked (needs the Banglish corpus first)

**Three things the audit found that change how you work here:**

1. **Label convention is `1 = correct`, `0 = hallucinated`** — an is-it-correct? flag, and the
   source files already use it, so nothing is flipped. Still load via `data/interim/`, which does
   the dedup and schema mapping.
2. **The pool is Bengali script, not Banglish** (0.96% Latin). It is the *base QA layer*, not the
   corpus. The Banglish generation step (guide §3.3) is the critical path.
3. **A substring rule scores 0.832 on the has-context half** — above the 0.80 project target,
   with no learning. Don't inherit that shortcut when generating the real corpus.

**Active focus — two tracks in parallel:**
- **Corpus (critical path):** answer the licence question → Banglish generation → M1 shortcut gate
- **Pipeline (unblocked now):** build the model ladder against `data/splits/bn_dev_benchmark/`

**M1 is done** (pilot gate passed at 0.505). **Next gate: M2** — two annotators label the same
100 blind items (`data/annotated/agreement_test_v1/`), Cohen's kappa >= 0.60 required. Below
that, fix the guidelines and rerun; don't touch the threshold.

---

## Setup

```bash
python -m venv .venv && source .venv/Scripts/activate     # Windows/Git Bash
pip install -r requirements.txt

# Or, on Kaggle/Colab:
pip install -q transformers datasets accelerate evaluate scikit-learn pandas numpy \
               scipy matplotlib seaborn gensim xgboost statsmodels torch torchtext \
               sentencepiece protobuf krippendorff sacremoses regex
pip install -q git+https://github.com/csebuetnlp/normalizer   # REQUIRED for Bangla(ish)BERT
```

## Commands

```bash
# Data pipeline — THESE FOUR EXIST AND WORK TODAY (all deterministic, seed 42)
python src/build_bn_pool.py         # raw -> interim -> splits/bn_dev_benchmark
python src/build_pilot.py           # interim -> generated/pilot_v1 (500 Banglish pairs)
python src/build_agreement_test.py  # pilot_v1 -> annotated/agreement_test_v1 (100 blind items)
python src/audit.py --data data/splits/bn_dev_benchmark  # all probes + structural checks

# THE GATE — must print < 0.60, exits non-zero if not
python src/audit.py --data data/generated/pilot_v1/pilot.jsonl --probe metadata
python src/audit.py --data data/interim/bn_pool.jsonl --strict

# Not yet written. Note the versioned output dir — never overwrite pilot_v1.
python src/generate.py   --config configs/generation.yaml --n 2000 --out data/generated/pilot_v2/
python src/preprocess.py --arm A --format F3 --in data/annotated/ --out data/splits/

# Training
python src/train_classical.py    --model ngram_logreg --seed 42
python src/train_transformer.py  --model csebuetnlp/banglishbert --arm A --format F3 --seed 42
python src/further_pretrain.py   --base bert-base-multilingual-cased --corpus data/raw/banglatlit_pt/

# Evaluation
python src/evaluate.py --checkpoint out/best --split dev      # any number of times
python src/evaluate.py --checkpoint out/best --split test     # EXACTLY ONCE, at M6
```

> `build_bn_pool.py` and `audit.py` are real and verified. The rest are the target interface.
> Every script takes `--seed` and appends a row to `results/experiment_log.csv`.

## Repo layout / entry points

| Path | Purpose |
|---|---|
| [docs/PRD.md](docs/PRD.md) | Requirements, gates, scope boundary — **the authority** |
| [docs/IMPLEMENTATION_GUIDE.md](docs/IMPLEMENTATION_GUIDE.md) | Recipes, hyperparameters, code sketches |
| `configs/schema.json` | Frozen record schema |
| `src/build_bn_pool.py` | ✅ Normalises + dedups the raw pool, builds grouped splits |
| `src/build_pilot.py` | ✅ Selects 500 pairs + drafts the Banglish transliteration |
| `src/build_agreement_test.py` | ✅ Builds the 100-item blind annotator-agreement test |
| `src/audit.py` | ✅ Probes + structural checks (see below) |
| `src/generate.py` | Paired faithful/hallucinated generation |
| `src/preprocess.py` | Arms A/B/C, formats F1/F2/F3, CMI |
| `src/train_classical.py` | N-gram, Skip-gram, BiRNN, BiLSTM(+attn) |
| `src/train_transformer.py` | Encoder fine-tuning |
| `src/further_pretrain.py` | MLM further pretraining |
| `src/evaluate.py` | Metrics, breakdowns, McNemar, bootstrap CI |
| [docs/ANNOTATION_GUIDELINES.md](docs/ANNOTATION_GUIDELINES.md) | Labelling rules — read before annotating (R4) |
| `requirements.txt` | Pinned deps. Tier 1 = data pipeline; Tier 2 = model ladder, commented out |
| `data/DATASET_AUDIT.md` | **Read first.** Full audit of the source pool + what's unmet |
| `data/generated/pilot_v1/` | ✅ 500 Banglish pairs, M1 gate passed. `review_sheet.csv` awaits your read-through |
| `data/annotated/agreement_test_v1/` | ✅ 100 blind items for the M2 kappa check, awaits 2 annotators |
| `data/raw/bn_qa_pool/` | 14 source `.jsonl`, verbatim. Has duplicates — don't load directly. |
| `data/interim/bn_pool.jsonl` | 56,480 cleaned, schema-mapped, label-corrected records |
| `data/splits/bn_dev_benchmark/` | Bengali-script pipeline benchmark (3112/531/538) — **not** the corpus |
| `data/splits/` | `train/dev/test.jsonl` — the real locked splits, created at M3 |
| `data/SOURCES.md` | Licence register — Wikipedia CC BY-SA 4.0 + BCS banks; 2 open questions |
| `results/experiment_log.csv` | Single source of truth for every run |

## Key facts to keep in your head

- `label`: **1 = correct/faithful**, **0 = incorrect/hallucinated** — an is-it-correct? flag.
  Positive class is *faithful*, the reverse of most papers. `probs[:, 1]` = P(correct).
- Splits: 60% has-context / 40% no-context · 50/50 class balance (±5%) · ~60/40 easy/hard
- Seeds: **42, 1337, 2024** — always all three, report mean ± std
- Fine-tune defaults: `lr=2e-5`, AdamW, `max_length=256`, `epochs=5`, `batch=32`, `warmup=0.1`,
  `weight_decay=0.01`, `fp16=True`, early stop on val loss (patience 2)
- FPT: MLM, 15% masking, `lr=1e-5`, batch 32, 5 epochs
- Targets: has-context **≥ 0.80**, no-context **≥ 0.60**, shortcut probe **< 0.60**, κ **≥ 0.60**
- Taxonomy: intrinsic `entity|numeric|relational|contradiction`; extrinsic `fabricated|overclaim`;
  negative `none`

---

## DO

- **Run the shortcut probe on the pilot before generating the full corpus.** It is a hard gate.
- **Log every run** to `results/experiment_log.csv` — including crashes and failures.
- **Set and log a seed** in anything that touches randomness.
- **Report has-context and no-context separately**, plus easy/hard and per-type breakdowns.
- **Sweep 3 arms × 3 formats** before touching architectures — it's the highest-ROI move (3–6 points).
- **Run `normalize()` after back-transliteration** for BanglaBERT/BanglishBERT — they were pretrained
  with it and skipping it costs points.
- **Record the licence in `data/SOURCES.md` before ingesting a source.** csebuetnlp = CC BY-NC-SA 4.0
  (non-commercial); Kaggle competition data may be benchmark-only, not redistributable.
- **Enforce length parity programmatically** (`tol=0.15`) and log the regeneration rate.
- **Use McNemar + bootstrap CI** before claiming one model beats another.
- **Investigate any macro-F1 > 0.95** on has-context as leakage.

## DON'T

- **DON'T touch `data/splits/test.jsonl`** except for the single sanctioned M6 evaluation.
- **DON'T weaken the shortcut probe** to make it pass — not the threshold, not the feature set, not
  the split. Fix the generation prompt and regenerate.
- **DON'T swap out a required model.** The ladder (N-gram → Skip-gram → BiRNN/BiLSTM → BERT-family,
  incl. BanglishBERT **and** MuRIL) is fixed by the course requirement. Adding is fine; replacing isn't.
- **DON'T use `AutoModelForMaskedLM` on BanglaBERT/BanglishBERT** — they're ELECTRA discriminators.
  FPT targets are mBERT and XLM-R.
- **DON'T apply csebuetnlp `normalize()` directly to Latin-script Banglish** — it's built for Bengali script.
- **DON'T rename, remove, or repurpose a schema field.** `cmi`, `script_condition`, and `error_span`
  are Phase 2 payload: populate them, never evaluate on them.
- **DON'T hand-curate a split** or re-roll a seed until numbers improve.
- **DON'T "fix" expected results.** Classical models at 0.50–0.65 and no-context well below
  has-context are correct outcomes, not bugs.
- **DON'T start Phase 2 work** — CMI degradation study, span-level evaluation, natural-vs-synthetic
  transfer, tokenizer fertility, LLM fine-tuning, paper writing. Flag it and stop.
- **DON'T commit API keys**, PII, or non-redistributable source data. Wikipedia content is
  CC BY-SA 4.0 — **share-alike**, so the released corpus likely inherits that licence. Never mix
  it into one file with CC BY-**NC**-SA sources like `csebuetnlp/squad_bn`.
- **DON'T load `data/raw/bn_qa_pool/` directly** — 8.9% duplicates, no schema. Use `data/interim/`.
- **DON'T flip labels anywhere.** `1 = correct` end to end, source files included.
- **DON'T promote `bn_dev_benchmark/` into `data/splits/`.** It's Bengali-script, unannotated, and
  carries a 0.832 substring shortcut. It validates the pipeline; it is not a result.

---

## When stuck

| Symptom | Check |
|---|---|
| Shortcut probe > 0.60 | Inspect `clf.coef_` → find the leaking feature → fix the prompt → regenerate |
| macro-F1 > 0.95 has-context | Leakage. Check for duplicate questions across splits; run the answer-only probe |
| Answer-only probe ≈ full model | Model isn't checking grounding — the answer text gives away the label |
| κ < 0.60 | Guidelines, not annotators. Write explicit rules for partial-correctness, non-entailed-but-true, and hedged answers |
| Model spread within 1–2 points | Expected (BanTH spread was 2.8). Use 3 seeds + McNemar; report ties honestly |
| CUDA OOM | `batch_size=16` + gradient accumulation. Don't silently change `max_length` mid-experiment |
| Score plateaued | Work the ROI list: arm×format sweep → FPT → hard negatives → ensemble → threshold |
| has-context score suspiciously high on the benchmark | Expected — the substring shortcut. Compare against the 0.832 rule baseline in `results/experiment_log.csv` |
| Per-class metrics look swapped | `1` is the **correct/faithful** class here, not hallucinated |
