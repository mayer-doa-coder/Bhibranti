# AGENT.md — BanglishHallu Phase 1

> Context and operating instructions for AI agents working in this repository.
> Source of truth: [docs/PRD.md](docs/PRD.md) and [docs/IMPLEMENTATION_GUIDE.md](docs/IMPLEMENTATION_GUIDE.md).
> If this file and those documents ever disagree, **the PRD wins** — and flag the drift.

---

## 1. Project overview

**BanglishHallu** builds a **hallucination detector for Bangla–English code-mixed text
("Banglish")** — romanised Bangla with interleaved English — in the educational domain
(BCS / SSC / HSC question answering).

The work is split into two phases. **This repository is Phase 1 only.**

| Phase | Deliverable |
|---|---|
| **Phase 1 (here)** | A validated, human-annotated corpus + a benchmarked model ladder + a working detector |
| Phase 2 (later) | Causal analysis of *how* code-mixing degrades detection; the publishable paper |

### 1.1 Core objectives (PRD §3.1)

| ID | Goal |
|---|---|
| G1 | Human-annotated Banglish hallucination corpus, **≥ 4,000 QA pairs** |
| G2 | Cover both **grounded (has-context)** and **closed-book (no-context)** conditions |
| G3 | Benchmark a full model ladder: lexical → embedding → recurrent → pretrained-contextual |
| G4 | **macro-F1 ≥ 0.80 has-context**, **≥ 0.60 no-context** |
| G5 | Pass the **shortcut audit** (corpus validity gate) |
| G6 | Reproducible pipeline + complete experiment log |

### 1.2 The single most important rule

> **A high score on an artifacted corpus is a failed deliverable.**

The metadata-only shortcut probe (§7) is a **blocking release gate**. Any agent that raises a score
by weakening, skipping, or reinterpreting that gate has actively damaged the project. Corpus
validity outranks every metric on every table.

### 1.3 Anti-goals — do not do these

- Do **not** maximise score at the expense of corpus validity.
- Do **not** chase state-of-the-art architectures. The model ladder is **fixed** by pedagogical
  requirement (this is also a course deliverable that must demonstrate N-gram, Skip-gram, RNN,
  LSTM, and BERT). Substituting a "better" model for a required one is a regression, not an
  improvement.
- Do **not** start Phase 2 work. See §11.

---

## 2. Tech stack and environment

### 2.1 Language and runtime

- **Python 3.10+**, PyTorch, Hugging Face `transformers`
- Target compute: **Kaggle (T4 ×2 / P100, 30h/week)** or **Google Colab**. Assume a free-tier GPU
  with limited VRAM and a hard weekly quota — see §2.4.

### 2.2 Dependencies

```bash
pip install -q transformers datasets accelerate evaluate
pip install -q scikit-learn pandas numpy scipy matplotlib seaborn
pip install -q gensim              # Skip-gram / Word2Vec
pip install -q torch torchtext
pip install -q sentencepiece protobuf
pip install -q krippendorff        # IAA (Krippendorff's alpha)
pip install -q sacremoses regex
pip install -q statsmodels         # McNemar's test
pip install -q xgboost

# REQUIRED for BanglaBERT / BanglishBERT — they were pretrained with this normaliser
pip install -q git+https://github.com/csebuetnlp/normalizer
```

Transliteration tooling (mix routes): `bnbphoneticparser` / `pyAvroPhonetic`,
`indic_transliteration`, `bntranslit`, or
`shadabtanjeed/mbart-banglish-to-bengali-transliteration` for the reverse (back-transliteration)
direction used by Arm B.

### 2.3 Environment rules

- `requirements.txt` must be **pinned** and kept current (PRD R5). Any new import means a new pin.
- **Seeds are fixed project-wide: `42`, `1337`, `2024`.** Never introduce a new seed casually;
  never leave a seed unset. Every run logs its seed (PRD R2).
- Every notebook must run top-to-bottom from a clean environment. If a notebook depends on hidden
  state, it is broken.
- API keys / tokens: environment variables only. Never commit a key, never inline one in a notebook.

### 2.4 GPU discipline (RK8)

- Base-size encoders only unless VRAM demonstrably allows large.
- `max_length=256` by default; 512 only when answers genuinely need it.
- `fp16=True`; `batch_size=32`, drop to 16 + gradient accumulation on OOM.
- Further pretraining takes 2–4 hours. Budget it; do not launch it speculatively.

---

## 3. Repository structure

```
banglishhallu/
├── data/
│   ├── raw/                  # sourced QA pairs, unmodified
│   ├── generated/            # LLM-generated candidate answers
│   ├── annotated/            # post-human-annotation
│   ├── splits/               # train.jsonl / dev.jsonl / test.jsonl  <- deterministic, script-made
│   └── SOURCES.md            # licence register — REQUIRED, one row per source
├── src/
│   ├── generate.py           # Step 4 — paired faithful/hallucinated generation
│   ├── audit.py              # Step 9 — shortcut probe + answer-only probe   <- THE GATE
│   ├── preprocess.py         # Step 6 — Arms A/B/C, formats F1/F2/F3, CMI
│   ├── train_classical.py    # Step 7.1 — N-gram, Skip-gram, BiRNN, BiLSTM
│   ├── train_transformer.py  # Step 7.2 — encoder fine-tuning
│   ├── further_pretrain.py   # Step 8 — MLM further pretraining
│   └── evaluate.py           # Step 12 — metrics, breakdowns, significance tests
├── notebooks/                # exploratory only; anything reusable moves to src/
├── configs/
│   └── schema.json           # frozen record schema
├── results/
│   ├── experiment_log.csv    # THE single source of truth — every run, including failures
│   └── tables/               # Tables 1–5
└── docs/
    ├── PRD.md
    ├── IMPLEMENTATION_GUIDE.md
    └── ANNOTATION_GUIDELINES.md   # written BEFORE annotation begins (R4)
```

**Convention:** logic lives in `src/`, notebooks orchestrate and visualise. A notebook cell that
defines a training loop is a code smell — move it to `src/` and import it.

---

## 4. Data contract

### 4.1 Record schema (frozen — `configs/schema.json`)

```json
{
  "id": "bh_000001",
  "source": "BEnQA | NCTB | custom",
  "subject": "physics",
  "level": "SSC | HSC | BCS",
  "condition": "has_context | no_context",
  "context": "…passage…",
  "question": "…",
  "reference_answer": "…",
  "candidate_answer": "…",
  "label": 0,
  "hallucination_type": "none | entity | numeric | relational | contradiction | fabricated | overclaim",
  "difficulty": "easy | hard",
  "generator_model": "…",
  "generation_seed": 42,
  "annotator_1": 0,
  "annotator_2": 0,
  "adjudicated": false,
  "script_condition": "banglish",
  "cmi": 0.42,
  "error_span": "…"
}
```

`label`: **`0` = faithful/correct**, **`1` = hallucinated**.

**Schema is frozen.** Adding a field is allowed; renaming, removing, or repurposing one is not — it
breaks Phase 2's inheritance. `script_condition`, `cmi`, and `error_span` are Phase 2 payload:
**populate them, never evaluate on them** in Phase 1.

### 4.2 Taxonomy (PRD §5.2) — adopt the field's vocabulary, do not invent one

| Class | Types |
|---|---|
| **Intrinsic** (has-context, faithfulness) | `entity`, `numeric`, `relational`, `contradiction` |
| **Extrinsic** (no-context, factuality) | `fabricated`, `overclaim` |
| **Negative** | `none` |

### 4.3 Corpus composition targets

| Property | Target | Tolerance |
|---|---|---|
| Total annotated pairs | 4,000 (floor 1,500) | — |
| has-context / no-context | 60 / 40 | script-verified |
| hallucinated / faithful | 50 / 50 | ±5% |
| easy / hard | ~60 / 40 | — |
| Train / Dev / Test | 3,000 / 500 / 500 | — |
| Human-written Banglish items | ≥ 500 | — |
| Human-written hallucinated answers held out in **test** | 100–200 | flagged |

### 4.4 Non-negotiable data rules

- **`data/SOURCES.md` records the licence of every source, before ingestion.** csebuetnlp datasets
  are **CC BY-NC-SA 4.0 (non-commercial research only)** — carry the licence forward and cite.
  Kaggle competition data may be **benchmark-only, not redistributable** — never build the released
  corpus on it.
- **No PII in released data** (D13). Scrub, then manually review.
- **Splits are deterministic and script-generated** (R6). Never hand-curate a split.
- **The test set is locked at M0 and opened exactly once, at M6** (E9, RK10). See §8.4.

---

## 5. Generation protocol

> **The rule that governs everything: correct and hallucinated answers must be indistinguishable
> except in factual content.**

Same generator model, same prompt template, same temperature, same requested length, same
formatting instructions. Both classes are produced **in a single call per item** so they cannot
diverge stylistically.

The prompt must forbid uncertainty markers in the hallucinated answer unless both answers carry
them, and must request JSON with keys `faithful`, `hallucinated`, `error_span`.

**Length control is programmatic, not prompted:**

```python
def length_ok(faithful, hallucinated, tol=0.15):
    lf, lh = len(faithful.split()), len(hallucinated.split())
    return abs(lf - lh) / max(lf, lh, 1) <= tol
```

Regenerate any failing pair. **Log the rejection rate** — above ~20% means the prompt needs
tightening, not that the filter needs loosening.

**Generation prompts are version-controlled** (R3). A prompt change is a commit with a rationale,
and it invalidates every item generated with the old prompt.

Difficulty: **easy** = wrong entity from a different domain (Newton → Shakespeare);
**hard** = plausible near-miss (9.8 → 9.6 m/s²; 1971 → 1972; a sibling concept).

---

## 6. Preprocessing arms and input formats

Build **three arms** and **three formats**; sweep all 9 on the best model (highest-ROI step).

| Arm | Definition |
|---|---|
| **A — Raw** | As-is. Strip URLs, collapse whitespace, remove PII. |
| **B — Normalised** | Back-transliterate Banglish → Bengali script, then csebuetnlp `normalize()`. |
| **C — Dual** | `banglish [SEP] bengali_script` — both views concatenated. |

> **Caveat:** csebuetnlp `normalize()` is built for **Bengali script**. Apply it *after*
> back-transliteration, never directly to Latin-script Banglish. For BanglaBERT/BanglishBERT it is
> **required** — skipping it degrades results, because those models were pretrained with it.

| Format | Template |
|---|---|
| **F1** plain | `question + " " + answer` |
| **F2** with context | `context + " " + question + " " + answer` |
| **F3** NLI-style | `[CLS] context [SEP] question + answer [SEP]` |

F3 usually wins on has-context — hallucination framed as textual entailment. Consider an XLM-R
checkpoint already fine-tuned on XNLI.

**CMI** (Das & Gambäck) — computed per item at preprocessing time:

```python
def cmi(tokens_lang_counts, n_tokens, n_language_independent):
    n = n_tokens - n_language_independent
    if n == 0:
        return 0.0
    return 100 * (1 - max(tokens_lang_counts.values()) / n)
```

---

## 7. The shortcut audit — the blocking gate

**Run this on the 500-item pilot BEFORE generating the full corpus (V2, M1).**

The metadata-only probe trains logistic regression on features containing **no content words**:
token count, char count, mean token length, latin-character ratio, punctuation counts, digit ratio,
caps ratio — over `candidate_answer` alone.

| Probe macro-F1 | Verdict |
|---|---|
| **< 0.55** | Clean. Proceed. |
| 0.55 – 0.60 | Borderline. Inspect `clf.coef_`, find the leaking feature. |
| **> 0.60** | **Artifacted. Fix the prompt and regenerate.** No exceptions. |

If it fails: inspect coefficients → identify the leaking feature → fix the generation prompt →
regenerate → re-probe. **Never** fix a failing probe by changing the probe, moving the threshold,
dropping features, or re-splitting until it passes.

**Three supporting checks:**

1. **Answer-only probe (V3).** Full encoder on `candidate_answer` alone. On has-context this must be
   *substantially* below the full-input model. If it isn't, the answer text gives away the label and
   the model is not checking grounding at all.
2. **Human-written holdout (V4).** 100–200 test items with human-written hallucinated answers. A
   collapse here means the model learned the generator's fingerprint. Report the gap either way.
3. **Suspicious scores (V5).** Any macro-F1 **> 0.95** on has-context is leakage until proven
   otherwise, and the investigation is documented.

---

## 8. Model ladder and evaluation

### 8.1 Required models (all Must-priority — none may be dropped)

| Tier | Models |
|---|---|
| **M1 N-gram** | TF-IDF (word 1–2 grams + char 3–5 grams) → LogReg, LinearSVC. Log **OOV rate**. |
| **M2 Skip-gram** | gensim Word2Vec `sg=1, dim=200, window=5, min_count=2`, averaged → LogReg, XGBoost. Log **vocab coverage**. |
| **M3 Recurrent** | BiRNN (1×256), BiLSTM (2×256, dropout 0.3), BiLSTM + additive attention. Skip-gram init. |
| **M4/M5 Encoders** | ≥ 5 fine-tuned, **must include `csebuetnlp/banglishbert` and `google/muril-base-cased`** |

Encoder pool: `csebuetnlp/banglishbert` (ELECTRA discriminator, needs the normaliser),
`csebuetnlp/banglabert`, `google/muril-base-cased`, `xlm-roberta-base`,
`bert-base-multilingual-cased`, `ai4bharat/IndicBERTv2-MLM-only`, Mixed-Distil-BERT.

Char n-grams matter disproportionately here: Banglish spelling variance fragments word-level
features, and character features partially recover it.

BanTH's spread across seven very different encoders was **2.8 points** (77.35 → 74.51). **Do not
expect architecture choice alone to transform the score** — preprocessing, input format, and
further pretraining move the needle more.

### 8.2 Fine-tuning defaults (BanTH-validated — do not tune away without a reason)

```
lr=2e-5, AdamW, max_length=256 (512 if needed), epochs=5, batch=32 (16 on OOM),
warmup_ratio=0.1, weight_decay=0.01, fp16=True,
early stopping on validation loss (patience 2), metric_for_best_model="macro_f1"
```

### 8.3 Further pretraining (highest-leverage single step)

MLM, 15% masking, `lr=1e-5`, batch 32, 5 epochs, on unlabeled transliterated Bangla
(**BanglaTLit-PT**, 243K texts; supplement with BanglishRev, MixSarc).

> **ELECTRA warning:** BanglaBERT and BanglishBERT are ELECTRA *discriminators*.
> `AutoModelForMaskedLM` will **not** load them cleanly. Use the paired generator checkpoint
> (`csebuetnlp/banglishbert_generator`) or the RTD objective. **mBERT and XLM-R are the
> straightforward FPT targets — start there.** BanTH found FPT helps mBERT most (+2.39) and
> slightly *hurt* XLM-R (−0.31). Expect ~2 points, not 20.

### 8.4 Evaluation requirements

| Rule | Detail |
|---|---|
| **Primary metric** | macro-F1 (also report accuracy, per-class P/R, AUROC, confusion matrix) |
| **Always split results** | has-context vs no-context; easy vs hard; per hallucination type; human-written vs generated |
| **Cross-validation** | 5-fold stratified on **train+dev** |
| **Seeds** | 3 seeds (42/1337/2024), report **mean ± std** |
| **Significance** | McNemar's test for pairwise comparison; bootstrap 95% CI (1000 resamples) on the headline |
| **Threshold** | Never default to 0.5. Sweep on **dev only**, apply to test **once**. |
| **Test set** | Opened exactly once, at M6 |

A 1-point gap across a single seed is **noise**. Do not claim model A beats model B without a
significance test.

### 8.5 Realistic targets — calibrate before reporting

| Setting | Realistic | Stop-and-audit |
|---|---|---|
| Has-context | **0.80 – 0.90** | > 0.95 |
| No-context | **0.60 – 0.75** | > 0.85 |
| Hard subset | 0.55 – 0.70 | — |
| Classical @ 4K | 0.50 – 0.65 | (expected — not a failure) |
| Zero-shot LLM | 0.60 – 0.72 | — |

External anchors: ViHallu best **84.80** (encoder-only baseline 32.83); BanTH best **77.36**;
MedHallu hard **0.625**; SHROOM-CAP Bengali zero-shot **~0.51**.

Classical models scoring 0.50–0.65 is **the correct result** — they have no pretraining to fall back
on. The no-context number being far below has-context is **correct, not a bug**. Report both plainly
rather than engineering them upward.

### 8.6 Score maximisation, in ROI order

1. **Input format × arm sweep** (9 configs) — often 3–6 points. *Do this first.*
2. **Further pretraining** — ~2 points on the right base model.
3. **Hard-negative balance** — if easy 0.90 / hard 0.55, add hard examples to training.
4. **Soft-voting ensemble** of top 3 encoders — reliably 1–3 points.
5. **Threshold tuning** on dev — 1–2 points.
6. Class weights / focal loss — only if balance drifts from 50/50.
7. Hyperparameter search — **lowest ROI, do not spend a week here.**

### 8.7 LLM reference point (M8)

Not a competitor — a **ceiling marker** and reviewer insurance. Zero-shot / few-shot on 300–500 test
items. Test four prompting strategies: non-explanatory, CoT, explanation, and **translation-based**
(prepend "translate the transliterated text into standard Bangla/English", then classify) — the last
topped BanTH's zero-shot results. Fine-tuned encoders should win by roughly 8 points; if a zero-shot
LLM wins by a wide margin, the training data is too small or too noisy.

---

## 9. Milestones (PRD §8)

| M# | Week | Milestone | Exit criteria |
|---|---|---|---|
| **M0** | 1 | Foundations | Repo, schema frozen, licences audited, seeds fixed, **test set designated and locked** |
| **M1** | 1 | **Pilot + validity gate** | 500 items; **shortcut probe < 0.60**; prompt finalised |
| **M2** | 2 | Annotation protocol | Guidelines written; 100-item pilot annotated; κ computed; guidelines revised |
| **M3** | 3 | Corpus complete | 4,000 items generated + annotated; IAA reported; splits created |
| **M4** | 4 | Pipeline working | Classical ladder + 1 transformer end-to-end; log populated |
| **M5** | 5 | Full benchmark | All models × 3 arms × 3 formats; 5-fold CV; 3 seeds |
| **M6** | 6 | **Phase 1 complete** | FPT + ensemble + threshold + LLM reference; Tables 1–5; final audit passed |

**M1 is a hard gate. No progression to M3 without a passing shortcut probe.**

**Annotation levels:** test = 100% human, double-annotated, adjudicated; dev = 100% human, single
annotator + spot check; train = generator label + human verification on a 20% sample (if that sample
shows > 5% label noise, verify more). Report **Cohen's κ** (2 annotators) or **Fleiss' κ** (3+), plus
Krippendorff's α as a robustness check. Floor: **κ ≥ 0.60**. Below 0.40 means the guidelines are
broken, not the annotators — the usual cause is under-defined edge cases (partially correct answers,
answers correct but not entailed by the context, hedged answers). Write explicit rules for each.

---

## 10. Coding standards

### 10.1 General

- Match the surrounding code's style, naming, and comment density.
- Pure functions in `src/`; no side effects at import time.
- Type hints on public functions. Docstrings state units and expected ranges.
- Data I/O is **JSONL** for records, **CSV** for logs and tables.
- No magic numbers for hyperparameters — they live in `configs/`.
- No hardcoded absolute paths.

### 10.2 Determinism (non-negotiable)

Every script that touches randomness sets and logs its seed:

```python
import random, numpy as np, torch

def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
```

Splits are produced by a script from a fixed seed — never by hand, never re-rolled.

### 10.3 The experiment log is mandatory

`results/experiment_log.csv`, created day one, appended for **every run including failures**:

```
run_id,date,model,input_format,preprocessing,seed,lr,batch,epochs,split,dev_macro_f1,test_macro_f1,notes
```

A run that isn't logged didn't happen. A failed run that isn't logged is a result silently deleted.
Deleting or rewriting past rows is prohibited — append a correction row instead.

### 10.4 Error handling

- **Fail loudly on data problems.** Schema violations, class-balance drift beyond tolerance, missing
  licence entries, and length-control failures raise — they never warn-and-continue.
- **Fail loudly on gate violations.** A failing shortcut probe halts the pipeline with a non-zero
  exit; it does not print a warning and proceed.
- Never silently drop records. Filtering is explicit, counted, and logged with the reason.
- Wrap external calls (LLM APIs, HF downloads) in bounded retries with backoff; log the failure and
  the retry count. Never swallow the exception.
- `try/except` must catch a specific exception. Bare `except:` and `except Exception: pass` are never
  acceptable in `src/`.
- OOM handling: reduce batch size and use gradient accumulation. Do **not** silently shrink
  `max_length` mid-experiment — that changes what the run measured.

### 10.5 Testing requirements

No heavy test framework is required, but these checks must exist as runnable scripts and must pass
before any milestone is called complete:

| Check | Asserts |
|---|---|
| **Schema validation** | Every record matches `configs/schema.json`; enums hold legal values; `label ∈ {0,1}`; `hallucination_type == "none"` iff `label == 0` |
| **Split integrity** | No `id` in two splits; splits regenerate identically from the seed; test ids match the locked M0 manifest |
| **Balance verification** | 60/40 condition split; 50/50 class balance ±5%; difficulty ~60/40 |
| **Length control** | Every generated pair passes `length_ok`; rejection rate logged |
| **PII scan** | No emails, phone numbers, or personal names in released fields |
| **Shortcut probe** | macro-F1 < 0.60 — **exit non-zero if violated** |
| **Leakage check** | No exact or near-duplicate `question` across train/test |
| **Determinism smoke test** | Same seed → same metrics on a small subset |

Any change to `src/preprocess.py`, `src/generate.py`, or split logic requires re-running schema
validation, split integrity, balance, and the shortcut probe.

---

## 11. Scope boundary — Phase 2 is off-limits

RK9 is a **high-likelihood** risk: Phase 2's experiments are more interesting than Phase 1's, and the
pull to start them early is real. Resist it. A half-built corpus with half-built experiments delivers
neither.

**Deferred — do not implement, do not "just prototype":**

| ID | Deferred |
|---|---|
| N1 | Script-controlled CMI degradation study (record `cmi`; don't run the experiment) |
| N2 | Transliteration normalisation as a *research finding* (it's a preprocessing arm here) |
| N3 | Natural vs. synthetic transfer experiment |
| N4 | Span-level hallucination annotation/evaluation (capture `error_span`; don't evaluate on it) |
| N5 | Tokenizer fertility analysis |
| N6 | LLM fine-tuning, RLHF, retrieval augmentation |
| N7 | Paper writing and venue submission |

Also out of scope entirely: multi-class/multi-label hallucination classification, hallucination
*mitigation* or correction, other language pairs, production deployment, domains beyond education.

If a task looks like Phase 2 work, **say so and stop** rather than building it.

---

## 12. Deliverables checklist (PRD §13 — Definition of Done)

- [ ] Corpus ≥ 4,000 annotated pairs with train/dev/test splits
- [ ] Inter-annotator κ ≥ 0.60 reported
- [ ] Shortcut probe < 0.60 macro-F1 on the **final** corpus
- [ ] All Must-priority models (M1–M5) trained and benchmarked
- [ ] Has-context macro-F1 ≥ 0.80
- [ ] No-context macro-F1 ≥ 0.60
- [ ] Results Tables 1–5 produced
- [ ] 5-fold CV and 3-seed results reported with variance
- [ ] Test set evaluated **exactly once**
- [ ] Experiment log complete, including failed runs
- [ ] Codebase reproducible from a clean environment
- [ ] Phase 1 technical report written, **including a limitations section**
- [ ] `data/SOURCES.md` complete with licences

**Required tables:** (1) main results by model tier, (2) difficulty breakdown, (3) per-hallucination-type
F1, (4) preprocessing arm × input format ablation, (5) shortcut audit results.

---

## 13. Working agreements for agents

1. **Read `docs/PRD.md` §3.2 before proposing any new capability.** If it's on the deferred list, stop.
2. **Never weaken a validity gate to make a number look better.** Report the honest number.
3. **Never touch the test set** outside the single sanctioned M6 evaluation.
4. **Log every run**, including the ones that crashed.
5. **Report negative and disappointing results plainly.** Classical models at 0.55 and no-context at
   0.62 are expected outcomes, not bugs to engineer away.
6. **Record the licence** whenever you introduce a data source, in `data/SOURCES.md`, before ingesting it.
7. When results look too good (> 0.95 has-context), **assume leakage and investigate** before celebrating.
8. Prefer the guide's validated recipes over invented ones; when you deviate, say why in the log's
   `notes` column.

---

## 14. Key references

- **BanTH** — arXiv 2410.13281 — closest methodological template; FPT recipe, baseline table,
  annotation guidelines (Appendix B)
- **ViHallu / DSC2025** — arXiv 2601.04711 — target paper structure
- **BenHalluEval** — arXiv 2605.31483 — the competing Bengali hallucination benchmark; know it in detail
- **BEnQA** — arXiv 2403.10900 — primary base QA source (parallel Bengali/English SSC & HSC)
- **MuRIL** — arXiv 2103.10730 · **BanglaBERT / BanglishBERT** — arXiv 2101.00204
- **BanglaTLit** — Findings of EMNLP 2024 (`2024.findings-emnlp.859`) — FPT corpus
- **MedHallu** — difficulty stratification, "not sure" class · **HaluEval / HalluLens** — taxonomy
- `github.com/EdinburghNLP/awesome-hallucination-detection` — living survey
