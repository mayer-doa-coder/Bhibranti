# CLAUDE.md — BanglishHallu Phase 1 Cheat Sheet

Bangla–English code-mixed ("Banglish") hallucination detection. Phase 1 = build the corpus + the
model ladder. Full context in [AGENT.md](AGENT.md); requirements in [docs/PRD.md](docs/PRD.md);
step-by-step recipes in [docs/IMPLEMENTATION_GUIDE.md](docs/IMPLEMENTATION_GUIDE.md).

---

## Current status

**Milestone M0 — Foundations. The repo is docs-only right now.**
Only `docs/`, `LICENSE`, `AGENT.md`, and `CLAUDE.md` exist. No `src/`, no `data/`, no `results/`.

**Active focus — the day-one checklist:**

- [ ] Scaffold the §"Repo layout" tree below
- [ ] `results/experiment_log.csv` with headers
- [ ] `data/SOURCES.md` with a licence row per source (before ingesting anything)
- [ ] Kaggle অলীকবচন rules/licence read and recorded (open questions Q1, Q2)
- [ ] `configs/schema.json` frozen
- [ ] Generation prompt template drafted with length + register constraints
- [ ] `docs/ANNOTATION_GUIDELINES.md` started
- [ ] Seeds fixed: **42, 1337, 2024**
- [ ] Test set designated and **locked** — not opened until Week 6

**Next gate: M1 — generate 500 pilot items, run the shortcut probe, require < 0.60 macro-F1.**
Nothing downstream of M1 starts until that probe passes.

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
# Data pipeline
python src/generate.py   --config configs/generation.yaml --n 500 --out data/generated/pilot.jsonl
python src/preprocess.py --arm A --format F3 --in data/annotated/ --out data/splits/

# THE GATE — must print < 0.60, exits non-zero if not
python src/audit.py --data data/generated/pilot.jsonl --probe metadata
python src/audit.py --data data/splits/test.jsonl --probe answer-only

# Training
python src/train_classical.py    --model ngram_logreg --seed 42
python src/train_transformer.py  --model csebuetnlp/banglishbert --arm A --format F3 --seed 42
python src/further_pretrain.py   --base bert-base-multilingual-cased --corpus data/raw/banglatlit_pt/

# Evaluation
python src/evaluate.py --checkpoint out/best --split dev      # any number of times
python src/evaluate.py --checkpoint out/best --split test     # EXACTLY ONCE, at M6
```

> These CLIs don't exist yet — treat them as the target interface when you write `src/`.
> Every script takes `--seed` and appends a row to `results/experiment_log.csv`.

## Repo layout / entry points

| Path | Purpose |
|---|---|
| [docs/PRD.md](docs/PRD.md) | Requirements, gates, scope boundary — **the authority** |
| [docs/IMPLEMENTATION_GUIDE.md](docs/IMPLEMENTATION_GUIDE.md) | Recipes, hyperparameters, code sketches |
| `configs/schema.json` | Frozen record schema |
| `src/generate.py` | Paired faithful/hallucinated generation |
| `src/audit.py` | **Shortcut + answer-only probes — the blocking gate** |
| `src/preprocess.py` | Arms A/B/C, formats F1/F2/F3, CMI |
| `src/train_classical.py` | N-gram, Skip-gram, BiRNN, BiLSTM(+attn) |
| `src/train_transformer.py` | Encoder fine-tuning |
| `src/further_pretrain.py` | MLM further pretraining |
| `src/evaluate.py` | Metrics, breakdowns, McNemar, bootstrap CI |
| `data/splits/` | `train.jsonl` (3000) / `dev.jsonl` (500) / `test.jsonl` (500, **locked**) |
| `data/SOURCES.md` | Licence register — required before ingest |
| `results/experiment_log.csv` | Single source of truth for every run |

## Key facts to keep in your head

- `label`: **0 = faithful**, **1 = hallucinated**
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
- **DON'T commit API keys**, PII, or non-redistributable source data.

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
