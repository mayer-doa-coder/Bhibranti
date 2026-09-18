# AGENT.md — Phase 1 (Bengali)

> Context and operating instructions for AI agents working in this repository.
> Source of truth: [docs/PRD.md](docs/PRD.md) and [docs/IMPLEMENTATION_GUIDE.md](docs/IMPLEMENTATION_GUIDE.md).
> If this file and those documents ever disagree, **the PRD wins** — and flag the drift.

---

## 1. Project overview

This project builds a **hallucination detector for Bengali question answering** in the
educational domain (BCS / SSC / HSC).

**Phase 1 — this repository — is Bengali only.** Bangla script in, Bangla script out. There is
no Banglish anywhere in Phase 1.

| Phase | Deliverable |
|---|---|
| **Phase 1 (here)** | A filtered, annotated **Bengali** corpus + a benchmarked model ladder + a working detector |
| Phase 2 (later) | The **Banglish** (romanised Bangla) study: same pipeline, code-mixed input, and how code-mixing degrades detection |

> The project originally started with Banglish and had to restart. A rule-based transliteration
> produced text a native speaker could not read, and the pilot built on it was discarded. Phase 2
> must use a real transliteration model, and a native speaker must read every batch before any
> annotation is built on it. See PRD §14.1.

### 1.1 Core objectives (PRD §3.1)

| ID | Goal |
|---|---|
| G1 | Human-annotated **Bengali** hallucination corpus, **≥ 4,000 QA pairs** — built (4,480) |
| G2 | Cover both **grounded (has-context)** and **closed-book (no-context)** conditions |
| G3 | Benchmark a full model ladder: lexical → embedding → recurrent → pretrained-contextual |
| G4 | **macro-F1 ≥ 0.80 on the has-context HARD subset**, **≥ 0.60 no-context** |
| G5 | Pass the **shortcut audit** (corpus validity gate) |
| G6 | Reproducible pipeline + complete experiment log |

### 1.2 The single most important rule

> **A high score on an artifacted corpus is a failed deliverable.**
>
> **And a has-context score reported without its string-matcher baseline is not a result.**
> The rule "if the answer appears in the passage, call it correct" already scores **0.823** on
> all usable has-context items, and **0.454** on the hard subset. Quote both, every time.
>
> **Load data only through `src/splits.py`.** It drops the 180 human-flagged pairs that are
> excluded from training and scoring (PRD §5.1d).
>
> **RAG is forbidden.** No retrieval at inference of any kind — see PRD §5.3a. **The corpus is
> final:** do not collect or generate more data. **QA only:** no fill-in-the-blank / cloze items.

The metadata-only shortcut probe (§7) is a **blocking release gate**. Any agent that raises a score
by weakening, skipping, or reinterpreting that gate has actively damaged the project. Corpus
validity outranks every metric on every table.

### 1.3 Anti-goals — do not do these

- Do **not** maximise score at the expense of corpus validity.
- Do **not** chase state-of-the-art architectures. The model ladder is **fixed** by pedagogical
  requirement (this is also a course deliverable that must demonstrate N-gram, Skip-gram, RNN,
  LSTM, and BERT). Substituting a "better" model for a required one is a regression, not an
  improvement. The ladder follows the course's `NLP Lab/` syllabus (Labs 1–5) — PRD §5.3b lists
  the 26 lab topics used and the 7 excluded, with reasons.
- Do **not** copy lab code onto Bengali text (English-only regex, NLTK stop words, Porter,
  WordNet). Use the Bengali recipes in guide §6 / PRD §5.3c.
- Do **not** use RAG or any inference-time retrieval (PRD §5.3a).
- Do **not** collect or generate more data — the corpus is final.
- Do **not** add fill-in-the-blank items — QA only; cloze is a different task.
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

**This is the ACTUAL tree, not a plan.** OK = exists and works today. TODO = not built yet.

```
project/
|-- data/
|   |-- raw/bn_qa_pool/                    OK   14 source .jsonl, verbatim, 62,084 records
|   |   \-- MANIFEST.md                    OK   per-file checksums + licence pointer
|   |-- interim/bn_pool.jsonl              OK   56,480 cleaned records (unfiltered pool)
|   |-- corpus/bn_v1/corpus.jsonl          OK   THE CORPUS - 4,480 pairs / 8,960 records
|   |-- splits/                            OK   the locked splits
|   |   |-- train.jsonl  dev.jsonl  test.jsonl      3,129 / 673 / 678 pairs (usable 3,067 / 619 / 614)
|   |   \-- _pool_sanity/                  OK   throwaway, git-ignored, never train on it
|   |-- annotated/agreement_test_v1/       OK   100 blind items for the M2 kappa check
|   |   |-- items_for_annotation_BLANK.csv      <- copy twice, one per annotator
|   |   \-- answer_key_DO_NOT_OPEN_YET.csv      <- opened only after both submit
|   |-- DATASET_AUDIT.md                   OK   read before touching data/
|   \-- SOURCES.md                         OK   licence register (open questions remain)
|-- src/
|   |-- build_bn_pool.py                   OK   raw -> interim
|   |-- build_corpus.py                    OK   pairs, filters, splits
|   |-- build_agreement_test.py            OK   corpus -> 100-item blind test
|   |-- score_agreement.py                 OK   Cohen's kappa, the M2 gate
|   |-- build_annotation_sheets.py       OK   M3 sheets (6 documented guards)
|   |-- validate_annotation.py           OK   checks filled sheets before merge
|   |-- audit.py                           OK   shortcut probes  <- THE BLOCKING GATE
|   |-- score_test_agreement.py            OK   Cohen's kappa on the full test split
|   |-- merge_annotation.py                OK   human sheets -> types + `excluded` flags
|   |-- splits.py                          OK   load_split() - the only data loader
|   |-- text_bn.py                         OK   Bengali cleaning/tokenizer/stop words/stemmer (Lab 1); --check
|   |-- features.py                        TODO M11 char n-gram LM, M12 edit distance + cosine (Labs 1-3)
|   |-- preprocess.py                      TODO input formats F1/F2/F3
|   |-- train_classical.py                 TODO M1 BoW/TF-IDF + NB/LR/SVM, M2 Skip-gram, M11, M12 (Labs 2-3)
|   |-- train_neural.py                    TODO M3 RNN/BiRNN/BiLSTM(+attn), M13 Transformer from scratch (Labs 4-5)
|   |-- train_transformer.py               TODO pretrained encoder fine-tuning (M4/M5)
|   |-- further_pretrain.py                TODO MLM further pretraining (mBERT / XLM-R only)
|   \-- evaluate.py                        TODO metrics, breakdowns, M14 word-order test, McNemar, bootstrap CI
|-- notebooks/                             exploratory only; reusable code moves to src/
|-- configs/schema.json                    OK   frozen record schema
|-- results/
|   |-- experiment_log.csv                 OK   every run, including failures
|   \-- tables/                            TODO Tables 1-8, produced at M5/M6
\-- docs/
    |-- PRD.md                             OK   requirements and gates - the authority
    |-- IMPLEMENTATION_GUIDE.md            OK   recipes and hyperparameters
    \-- ANNOTATION_GUIDELINES.md           OK   written before annotation began (R4)
```

**Where the project is right now:** M1 done (corpus built, gate passing at 0.534).
**M2 PASSED** - kappa = 0.717 on 100 blind items (see agreement_test_v1/IAA_REPORT.md).
**M3 DONE** - all 5,310 judgements in `data/annotated/round1/` are done (test kappa = 0.865),
253 rows adjudicated, merged into corpus + splits (strict audit passes). 180 human-flagged
pairs are excluded from training and scoring (PRD 5.1d) - usable: train 3,067 / dev 619 /
test 614 pairs. **M4 (model ladder) is next.** D8 is narrowed to
test + dev + the 20% train spot-check (PRD 5.1c); the other train records stay 'unlabeled' with
`type_source = outside_train_sample` by design.

**Convention:** logic lives in `src/`, notebooks orchestrate and visualise. A notebook cell that
defines a training loop is a code smell - move it to `src/` and import it. Every `build_*.py` is
deterministic at seed 42 and reproduces byte-identical output.

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
  "label": 1,
  "hallucination_type": "none | entity | numeric | relational | contradiction | fabricated | overclaim",
  "difficulty": "easy | hard",
  "generator_model": "…",
  "generation_seed": 42,
  "annotator_1": 0,
  "annotator_2": 0,
  "adjudicated": false,
  "script_condition": "bengali",
  "cmi": 0.42,
  "error_span": "…"
}
```

### Label convention — PROJECT-WIDE, NON-NEGOTIABLE (PRD §5.1a)

```
label = 1  ->  CORRECT / FAITHFUL     (no hallucination)
label = 0  ->  INCORRECT / HALLUCINATED
```

An **is-it-correct?** flag. This matches the ingested source data natively, so **no flip is
applied anywhere in the pipeline**. `hallucination_type == "none"` exactly when `label == 1`.

The **positive class (1) is faithful**, not hallucinated — the reverse of much of the
literature. Macro-F1 is unaffected (symmetric under a global flip); per-class precision/recall
must say which class is meant. `probs[:, 1]` is P(correct).

**Schema is frozen.** Adding a field is allowed; renaming, removing, or repurposing one is not — it
breaks Phase 2's inheritance. `script_condition`, `cmi`, and `error_span` are Phase 2 payload:
**populate them, never evaluate on them** in Phase 1.

### 4.2 Taxonomy (PRD §5.2) — adopt the field's vocabulary, do not invent one

| Class | Types |
|---|---|
| **Intrinsic** (has-context, faithfulness) | `entity`, `numeric`, `relational`, `contradiction` |
| **Extrinsic** (no-context, factuality) | `fabricated`, `overclaim` |
| **Faithful** (`label == 1`) | `none` |

### 4.3 Corpus composition — as built (targets met)

| Property | Target | Actual |
|---|---|---|
| Total pairs | ≥ 4,000 | **4,480** (8,960 records); **4,300 usable** after 180 human-flagged pairs were excluded (PRD §5.1d) |
| has-context / no-context | 60 / 40 | 2,688 / 1,792 pairs |
| hallucinated / faithful | 50 / 50 | exact, in every split, before and after exclusion |
| easy / hard (pairs) | measured, not targeted | has-context 1,761 / 927 · no-context 1,385 / 407 |
| Train / Dev / Test (pairs) | 70 / 15 / 15, by pair | 3,129 / 673 / 678 → usable **3,067 / 619 / 614** |
| Human-written hallucinated answers | — | none exist: every record is `llm_generated` (a limitation to report) |

### 4.4 Non-negotiable data rules

- **`data/SOURCES.md` records the licence of every source, before ingestion.** csebuetnlp datasets
  are **CC BY-NC-SA 4.0 (non-commercial research only)** — carry the licence forward and cite.
  Kaggle competition data may be **benchmark-only, not redistributable** — never build the released
  corpus on it.
- **No PII in released data** (D13). Scrub, then manually review.
- **Splits are deterministic and script-generated** (R6). Never hand-curate a split.
- **The test set is locked at M0 and opened exactly once, at M6** (E9, RK10). See §8.4.

### 4.5 Current data state — read `data/DATASET_AUDIT.md` before touching data

A 62,084-record QA pool has been ingested. Three facts govern how it may be used:

| Fact | Consequence |
|---|---|
| **Source labels already match the project convention** (`1 = correct`) | No flip anywhere. Still load via `data/interim/bn_pool.jsonl` — that stage does the dedup and schema mapping. |
| **The pool is Bengali script** (0.96% Latin) | Correct and expected — Phase 1 is Bengali. `src/build_corpus.py` filters it to the 4,480-pair corpus. |
| **A substring rule scores 0.823 macro-F1 on has-context** (usable data; 0.812 before exclusion) | Correct answers are verbatim spans of the passage far more often than wrong ones. Handled by the `difficulty` field, not by deletion: on the **hard** subset the same rule scores 0.454. Always report both. |

The PRD's own metadata gate (V1) **passes** on this pool at 0.453–0.506 — notable because the
pool *was* LLM-constructed, so the classic generation artifact could have been there and isn't.

**Sources:** Bengali Wikipedia (**CC BY-SA 4.0** — attribution + share-alike, so the released
corpus likely inherits it) and BCS question banks (terms unresolved). QA pairs were built from
those texts with LLM assistance, recorded per record as `provenance: llm_generated`. Keep that
field accurate — PRD D7/V4 and Phase 2's synthetic-vs-natural claim both depend on being able to
separate generated from human-written items. Full register: `data/SOURCES.md`.

`data/splits/{train,dev,test}.jsonl` are the real Phase 1 splits, built by
`src/build_corpus.py` from `data/corpus/bn_v1/corpus.jsonl`. They are Bengali script,
QA only (no fill-in-the-blank), grouped by pair, and deterministic at seed 42. Annotation is merged
into them (`src/merge_annotation.py`); load them only with `src/splits.py`.

`data/splits/_pool_sanity/` is a throwaway artifact of `build_bn_pool.py`, used only to
check that script still runs. It is git-ignored. Never train on it and never report from it.

## 5. Generation — not done in Phase 1

**Phase 1 generates no data. The corpus is final** (PRD §5.1). The source pool's QA pairs were
built with LLM assistance before this project; the model and prompts are unrecorded (PRD Q4).

The principle still matters when *reading* results: correct and hallucinated answers must differ
only in factual content. That is what the metadata probe checks (§7, currently 0.534 / 0.514), and
why answer-only models are reported beside the V3 probe.

Difficulty is **measured** by `src/build_corpus.py`, not generated: has-context **hard** = the
string matcher cannot separate the pair's two answers; no-context **hard** = the wrong answer is a
near-miss (character similarity ≥ 0.60).

---

## 6. Preprocessing arms and input formats

**Two arms × three formats** = 6 configurations, swept on the best pretrained encoder (M9, the
highest-ROI step). The non-pretrained rungs use the Bengali cleaning, tokenizer and M10 variants in
guide §6 instead (PRD §5.3c: never copy the English lab code).

| Arm | Definition |
|---|---|
| **A — Raw** | Guide §6 cleaning only (NFC, citation marks, whitespace). Digits and negation kept. |
| **B — Normalised** | A + csebuetnlp `normalize()`. Required for BanglaBERT. |
| ~~C — Dual~~ | **Phase 2 only.** There is no second script view in Phase 1. |

> **Caveat:** csebuetnlp `normalize()` is built for **Bengali script**. Apply it *after*
> Bengali script, which is all of Phase 1. For BanglaBERT it is
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

**Rerun after every change to the corpus or its annotation (V2).** Current: **PASS** — 0.534 on all
records, 0.514 on usable records; both must stay below 0.60. `src/audit.py` checks both.

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

| Tier | Models | Lab |
|---|---|---|
| **M1 Sparse n-gram** | BoW and TF-IDF (word 1–2 + char 3–5 grams, one block per record part) → **Naive Bayes** (α = 1), LogReg, LinearSVC. Log **OOV rate**. | 2, 3 |
| **M2 Skip-gram** | gensim Word2Vec `sg=1, dim=200, window=5, min_count=2` on the train split; **mean** and **TF-IDF-weighted mean** → LogReg, XGBoost. Log **vocab coverage**. | 3 |
| **M3 Recurrent (PyTorch)** | vanilla RNN (1×256), BiRNN (1×256), BiLSTM (2×256, dropout 0.3), BiLSTM + **dot-product attention** (`torch.bmm`). Skip-gram init via `nn.Embedding.from_pretrained`; packed sequences; single logit + `BCEWithLogitsLoss`. | 4 |
| **M4/M5 Encoders** | ≥ 5 fine-tuned, **must include `csebuetnlp/banglabert` and `google/muril-base-cased`** | — |
| **M10 Preprocessing ablations** | Bengali clean + tokenize (default); + stop words (negation kept); + stemming (never strips negation); V2-demo with negation removed | 1 |
| **M11 Char n-gram LM** | Laplace-smoothed; (a) passage-conditioned answer score, (b) class-conditional classifier (answer-only → report beside V3) | 2, 3 |
| **M12 Similarity features** | normalised Levenshtein + Skip-gram cosine, answer vs **its own** passage/question → LogReg; defines the V6 fuzzy string-matcher baseline | 1, 3 |
| **M13 Transformer from scratch** | d_model 128, 4 heads, 2 layers, sine/cosine positional encoding, padding mask, masked mean pooling | 5 |
| **M14 Word-order diagnostic** | shuffle token order on dev, report Δ macro-F1 per model and per type | 3 |

Excluded lab topics (PRD §5.3b): lemmatization, spelling correction, text generation / Shannon
game, LSTM LM sampling, POS tagging, Seq2Seq translation, word analogies. Pretrained fastText
vectors: not used (decision 2026-09-17).

Full recipes: guide §6 (preprocessing) and §7.1–7.1d (models); lab → code map in guide Appendix B.

Encoder pool: `csebuetnlp/banglabert` (ELECTRA discriminator, needs the normaliser; primary for Phase 1),
`csebuetnlp/banglabert`, `google/muril-base-cased`, `xlm-roberta-base`,
`bert-base-multilingual-cased`, `ai4bharat/IndicBERTv2-MLM-only`, Mixed-Distil-BERT.

Char n-grams matter disproportionately here: Bangla inflection and compounding fragment word-level
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
(Phase 1: unlabeled Bengali text. BanglaTLit-PT / BanglishRev / MixSarc are **Phase 2** corpora.)

> **ELECTRA warning:** BanglaBERT and BanglishBERT are ELECTRA *discriminators*.
> `AutoModelForMaskedLM` will **not** load them cleanly. Use the paired generator checkpoint
> (`csebuetnlp/banglishbert_generator`) or the RTD objective. **mBERT and XLM-R are the
> straightforward FPT targets — start there.** BanTH found FPT helps mBERT most (+2.39) and
> slightly *hurt* XLM-R (−0.31). Expect ~2 points, not 20.

### 8.4 Evaluation requirements

| Rule | Detail |
|---|---|
| **Primary metric** | macro-F1 (also report accuracy, per-class P/R, AUROC, confusion matrix) |
| **Always split results** | has-context vs no-context; easy vs hard; per subject; per hallucination type (**dev/test only** — PRD 5.1c); human-written vs generated |
| **Cross-validation** | 5-fold stratified on **train+dev** |
| **Seeds** | 3 seeds (42/1337/2024), report **mean ± std** |
| **Significance** | McNemar's test for pairwise comparison; bootstrap 95% CI (1000 resamples) on the headline |
| **Threshold** | Never default to 0.5. Sweep on **dev only**, apply to test **once**. |
| **Baselines beside every has-context score** | exact string matcher (0.823 / 0.454 hard, usable data) and fuzzy string matcher (V6) |
| **Ablations and diagnostics** | M10 preprocessing grid and M14 word-order test on **dev only** |
| **Test set** | Opened exactly once, at M6 |

A 1-point gap across a single seed is **noise**. Do not claim model A beats model B without a
significance test.

### 8.5 Realistic targets — calibrate before reporting

| Setting | Realistic | Stop-and-audit |
|---|---|---|
| Has-context | **0.80 – 0.90** | > 0.95 |
| No-context | **0.60 – 0.75** | > 0.85 |
| Hard subset | 0.55 – 0.70 | — |
| Classical @ 4K (M1, M2, M11) | 0.50 – 0.65 | (expected — not a failure) |
| Recurrent (M3) | 0.55 – 0.70 | — |
| Transformer from scratch (M13) | near M3, well below pretrained | above pretrained → audit |
| Zero-shot LLM | 0.60 – 0.72 | — |

External anchors: ViHallu best **84.80** (encoder-only baseline 32.83); BanTH best **77.36**;
MedHallu hard **0.625**; SHROOM-CAP Bengali zero-shot **~0.51**.

Classical models scoring 0.50–0.65 is **the correct result** — they have no pretraining to fall back
on. The no-context number being far below has-context is **correct, not a bug**. Report both plainly
rather than engineering them upward.

### 8.6 Score maximisation, in ROI order

1. **Input format × arm sweep** (6 configs) — often 3–6 points. *Do this first.*
2. **Further pretraining** — ~2 points on the right base model.
3. **Hard-negative balance** — if easy 0.90 / hard 0.55, add hard examples to training.
4. **Soft-voting ensemble** of top 3 encoders — reliably 1–3 points.
5. **Threshold tuning** on dev — 1–2 points.
6. Class weights / focal loss — only if balance drifts from 50/50.
7. Hyperparameter search — **lowest ROI, do not spend a week here.**

### 8.7 LLM reference point (M8)

Not a competitor — a **ceiling marker** and reviewer insurance. Zero-shot / few-shot on 300–500
items; prompts developed on dev, test scored only inside the single M6 evaluation. Test three
prompting strategies: non-explanatory, CoT, explanation. (BanTH's translation-based prompt is for
transliterated text — Phase 2 only.)

**No retrieval (PRD §5.3a):** the prompt holds only the record's own question, passage and answer.
Few-shot examples are **one fixed set** drawn once from train (seed 42) — never chosen per item by
similarity. No browsing or search tools. Fine-tuned encoders should win by roughly 8 points; if a zero-shot
LLM wins by a wide margin, the training data is too small or too noisy.

---

## 9. Milestones (PRD §8)

| M# | Milestone | Exit criteria | Status |
|---|---|---|---|
| **M0** | Foundations | Repo, schema frozen, seeds fixed, source pool ingested and audited | ✅ |
| **M1** | Corpus built + validity gate | 4,480 pairs, QA only, split by pair; **shortcut probe < 0.60** | ✅ 0.534 |
| **M2** | Annotation protocol | Guidelines written; 100-item blind pilot; **κ ≥ 0.60**; guidelines revised | ✅ κ 0.717 |
| **M3** | Corpus annotated | Test double-annotated, dev + 20% train single; adjudicated; merged; flagged pairs excluded | ✅ κ 0.865; 4,300 usable pairs |
| **M4** | Model ladder trained | M1–M5, M10–M13 run and logged; V6 fuzzy baseline measured | ⬜ **next** |
| **M5** | Sweeps and extensions | M6 FPT, M7 ensemble, M8 LLM reference, M9 formats, M14 word-order test | ⬜ |
| **M6** | **Phase 1 complete** | Test scored **once**; Tables 1–8; final audit passed; report with limitations | ⬜ |

**The shortcut gate (§7) is a hard gate:** rerun it after any corpus change; nothing is trained on a
corpus that fails it.

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
| **Schema validation** | Every record matches `configs/schema.json`; enums hold legal values; `label ∈ {0,1}`; `hallucination_type == "none"` iff `label == 1` |
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

## 11. Scope boundary — Phase 2 (Banglish) is off-limits

RK9 is a **high-likelihood** risk: Phase 2's experiments are more interesting than Phase 1's, and the
pull to start them early is real. Resist it. A half-built corpus with half-built experiments delivers
neither.

**Deferred — do not implement, do not "just prototype":**

| ID | Deferred |
|---|---|
| N1 | Script-controlled CMI degradation study (record `cmi`; don't run the experiment) |
| N2 | Transliteration normalisation as a *research finding* |
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

- [x] Corpus ≥ 4,000 annotated pairs with train/dev/test splits (4,480; 4,300 usable)
- [x] Inter-annotator κ ≥ 0.60 reported (0.717 pilot, 0.865 full test)
- [x] Shortcut probe < 0.60 macro-F1 on the **final** corpus (0.534 all / 0.514 usable)
- [ ] All ladder models (M1–M14) trained and benchmarked; every lab topic in PRD §5.3b covered
- [ ] Has-context **hard subset** macro-F1 ≥ 0.80
- [ ] No-context macro-F1 ≥ 0.60
- [ ] Results Tables 1–8 produced (guide §12.4)
- [ ] 5-fold CV and 3-seed results reported with variance
- [ ] Test set evaluated **exactly once**
- [ ] Experiment log complete, including failed runs
- [ ] Codebase reproducible from a clean environment
- [ ] Phase 1 technical report written, **including a limitations section**
- [ ] `data/SOURCES.md` complete with licences

**Required tables:** (1) main results by model tier, in lab order, (2) difficulty breakdown,
(3) per-hallucination-type F1, (4) normalisation × input format ablation, (5) shortcut audit
results, (6) M10 preprocessing ablation, (7) M14 word-order diagnostic, (8) per-subject F1.

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
