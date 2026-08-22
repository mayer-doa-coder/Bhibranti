# BanglishHallu — Phase 1 Implementation Guide

**Scope:** Build a working Bangla–English code-mixed hallucination detector and maximise its detection score.
**Explicitly out of scope for Phase 1:** the script-controlled CMI degradation study, the normalisation ablation as a *research finding*, span-level annotation, and the paper's causal argument. Those are Phase 2.

**Version:** 1.0
**Last verified:** 23 August 2026

---

## 0. What "success" means in Phase 1

You are done when all five of these hold:

| # | Criterion | Target |
|---|---|---|
| C1 | Corpus built, annotated, IAA reported | ≥ 4,000 pairs, Cohen's κ ≥ 0.60 |
| C2 | Best model on **has-context** split | Macro-F1 ≥ 0.80 |
| C3 | Best model on **no-context** split | Macro-F1 ≥ 0.60 |
| C4 | **Shortcut audit passes** | Metadata-only classifier < 0.60 macro-F1 |
| C5 | Full experiment log exists | Every run recorded with seed + config |

C4 is not optional. A high score that fails C4 is worthless — it means your model learned generation artifacts, not hallucination. Read Step 9 before you generate a single data point.

---

## 1. Environment setup

### 1.1 Hardware

Kaggle (free T4 x2 / P100, 30h/week GPU quota) or Google Colab is sufficient for everything in Phase 1. Base-size encoders fine-tune on ~5K examples in 10–25 minutes. Further pretraining (Step 8) takes 2–4 hours.

### 1.2 Dependencies

```bash
pip install -q transformers datasets accelerate evaluate
pip install -q scikit-learn pandas numpy scipy matplotlib seaborn
pip install -q gensim              # Skip-gram / Word2Vec
pip install -q torch torchtext
pip install -q sentencepiece protobuf
pip install -q krippendorff        # IAA
pip install -q sacremoses regex

# BanglaBERT / BanglishBERT normalisation pipeline (REQUIRED for those models)
pip install -q git+https://github.com/csebuetnlp/normalizer
```

### 1.3 Repository layout

```
banglishhallu/
├── data/
│   ├── raw/                 # scraped / sourced QA pairs
│   ├── generated/           # LLM-generated candidate answers
│   ├── annotated/           # post-human-annotation
│   └── splits/              # train.jsonl / dev.jsonl / test.jsonl
├── src/
│   ├── generate.py          # Step 4
│   ├── audit.py             # Step 9 — shortcut detection
│   ├── preprocess.py        # Step 6
│   ├── train_classical.py   # Step 7
│   ├── train_transformer.py # Step 7
│   ├── further_pretrain.py  # Step 8
│   └── evaluate.py
├── notebooks/
├── results/
│   └── experiment_log.csv   # THE single source of truth
└── configs/
```

Create `results/experiment_log.csv` on day one with this header and append **every** run, including failures:

```
run_id,date,model,input_format,preprocessing,seed,lr,batch,epochs,split,dev_macro_f1,test_macro_f1,notes
```

---

## 2. Data schema

Fix this before generating anything. Every record:

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
  "hallucination_type": "none | entity | numeric | relational | fabricated | contradiction",
  "difficulty": "easy | hard",
  "generator_model": "…",
  "generation_seed": 42,
  "annotator_1": 0,
  "annotator_2": 0,
  "adjudicated": false,
  "script_condition": "banglish",
  "cmi": 0.42
}
```

`label`: `0` = faithful/correct, `1` = hallucinated.

Two fields deserve early attention even though they are Phase 2 payload: `script_condition` and `cmi`. Recording them now costs nothing and saves you from rebuilding the corpus later.

**CMI (Code-Mixing Index)** — compute per Das & Gambäck's formulation:

```python
def cmi(tokens_lang_counts, n_tokens, n_language_independent):
    """tokens_lang_counts: dict of language -> token count"""
    n = n_tokens - n_language_independent
    if n == 0:
        return 0.0
    max_wi = max(tokens_lang_counts.values())
    return 100 * (1 - max_wi / n)
```

---

## 3. Sourcing the base QA pairs

### 3.1 Recommended sources (in priority order)

| Source | What it gives you | Access |
|---|---|---|
| **BEnQA** | ~5K parallel Bengali/English SSC & HSC science exam questions (factual, application, reasoning types) | `github.com/sheikhshafayat/BEnQA` |
| **NCTB-QA** | 87,805 educational QA pairs over 10,070 contexts, 42.75% unanswerable | arXiv 2603.05462 |
| **BanglaRQA** | 14,889 QA pairs, 3,000 contexts | Public |
| **squad_bn** (csebuetnlp) | Bengali SQuAD 2.0 + TyDiQA translations | `csebuetnlp/squad_bn` on HF |
| Your own BCS/SSC/HSC scrape | Full control, novel | — |

BEnQA is the strongest starting point: it is already parallel Bengali–English at exactly your target education level, which means generating the Banglish condition is a transliteration step rather than a translation step.

**Unanswerable questions matter.** NCTB-QA's 42.75% unanswerable slice is exactly where extrinsic hallucination lives — a model that fabricates an answer to an unanswerable question *is* the phenomenon you're detecting. Include some.

### 3.2 Licence check — do this first

Before ingesting anything:

- **csebuetnlp datasets** (`squad_bn`, `xnli_bn`) are **CC BY-NC-SA 4.0 — non-commercial research only**. Fine for your purposes, but you must carry the licence forward and cite.
- **Kaggle competition data** (অলীকবচন / bengali-hallucination): read the rules tab. Competition-only licences prohibit redistribution. If so, you may use it for internal benchmarking but **cannot ship it in your released dataset**.
- Scraped exam questions: check the board/publisher terms.

Record the licence of every source in a `data/SOURCES.md` file now. Reviewers and journal editors will ask.

### 3.3 Producing the Banglish condition

Three routes, use all three and mix:

1. **Native writers.** Ask 3–5 Bangladeshi students to rewrite items the way they'd actually type them. Highest quality, most naturalistic — this is what distinguishes you from synthetic benchmarks. Aim for at least 500 items via this route.
2. **Transliteration tooling** applied to the Bengali-script version:
   - `bnbphoneticparser` / `pyAvroPhonetic` — Avro-style phonetic parsers
   - `indic_transliteration` (Python)
   - `bntranslit`
   - `shadabtanjeed/mbart-banglish-to-bengali-transliteration` (reverse direction, useful for Step 6)
   - Reference dataset: `SKNahin/bengali-transliteration-data` on HF
3. **LLM rewriting** with explicit instruction to preserve natural Banglish spelling variance.

Then have a native speaker spot-check ~10% of routes 2 and 3. Automated transliteration produces unnaturally consistent spelling; real Banglish is inconsistent, and that inconsistency is the actual difficulty.

**Target sizes:**

| Split | Count | Notes |
|---|---|---|
| Train | ~3,000 | 60% has-context, 40% no-context |
| Dev | ~500 | Same stratification |
| Test | ~500 | **100% human-verified**, stratified |
| **Total** | **~4,000** | Absolute floor 1,500; ceiling of usefulness ~10,000 |

Class balance: 50/50 hallucinated/faithful. Balanced classes make macro-F1 interpretable and prevent a majority-class classifier from looking good. (BanTH's authors flagged exactly this: their 72.73% non-hate majority meant a dummy classifier beat GPT-3.5.)

---

## 4. Generating hallucinated candidates

### 4.1 The generation protocol

**The rule that governs everything: correct and hallucinated answers must be indistinguishable except in factual content.**

```
Same generator model.
Same prompt template structure.
Same temperature.
Same requested output length.
Same formatting instructions.
Only the factual content differs.
```

Concretely — generate both classes in a single call per item:

```
System: You are creating evaluation data for a hallucination detection benchmark.

Given the context, question, and reference answer below, produce TWO answers:

  A) A FAITHFUL answer — factually correct and grounded in the context.
  B) A HALLUCINATED answer — fluent and plausible, but containing exactly one
     factual error of type: {TYPE}

CONSTRAINTS (apply to BOTH A and B):
  - Both must be between {L_min} and {L_max} tokens.
  - Both must use the same register, formatting, and hedging level.
  - Both must be written in Bangla–English code-mixed style matching the question.
  - Neither may contain phrases like "I think", "possibly", "it may be" unless
    both contain them.
  - Do not signal uncertainty in B. B must sound as confident as A.

Return JSON: {"faithful": "...", "hallucinated": "...", "error_span": "..."}
```

The `error_span` field is a free win — it costs one extra JSON key now and gives you span-level supervision for Phase 2.

### 4.2 Hallucination taxonomy

Adopt the field's vocabulary rather than inventing one. Map to the intrinsic/extrinsic split used by ViHallu and HalluLens:

**Intrinsic (has-context / faithfulness):**
| Type | Definition |
|---|---|
| `entity` | Wrong named entity substituted (person, place, organisation) |
| `numeric` | Wrong number, date, quantity, or unit |
| `relational` | Correct entities, wrong relation between them |
| `contradiction` | Directly contradicts a statement in the context |

**Extrinsic (no-context / factuality):**
| Type | Definition |
|---|---|
| `fabricated` | Content with no grounding in any source; invented fact |
| `overclaim` | Answers a genuinely unanswerable question with false confidence |

### 4.3 Easy vs. hard difficulty

Generate both. MedHallu found the best model reached only 0.625 F1 on their "hard" category, and that harder-to-detect hallucinations are semantically closer to ground truth. A difficulty breakdown makes your results table far more informative than a single number.

- **Easy:** wrong entity from a different domain (Newton → Shakespeare)
- **Hard:** plausible near-miss (9.8 m/s² → 9.6 m/s²; 1971 → 1972; a sibling concept)

Target ~60% easy / 40% hard.

### 4.4 Length control — enforce it programmatically

Do not trust the prompt. Verify:

```python
def length_ok(faithful, hallucinated, tol=0.15):
    lf, lh = len(faithful.split()), len(hallucinated.split())
    return abs(lf - lh) / max(lf, lh, 1) <= tol
```

Regenerate any pair that fails. Log the rejection rate — if it's above ~20%, your prompt needs tightening.

---

## 5. Annotation

### 5.1 Protocol

- **2 annotators minimum**, both native Bangla speakers comfortable with Banglish. 3 is better.
- Write an explicit guidelines document *before* annotation begins. BanTH's Appendix B is an excellent template to imitate — it gives per-category definitions with worked bilingual examples.
- Run a **pilot on 100 items**, compute agreement, resolve disagreements, revise guidelines, *then* annotate the rest.
- A domain expert (you, or your supervisor) adjudicates disagreements.
- Annotators label: `label` (0/1), and for label=1, `hallucination_type`.

### 5.2 Inter-annotator agreement

Report **Cohen's κ** for 2 annotators, **Fleiss' κ** for 3+, and Krippendorff's α as a robustness check.

```python
from sklearn.metrics import cohen_kappa_score
kappa = cohen_kappa_score(df.annotator_1, df.annotator_2)
```

Interpretation floor: **κ ≥ 0.60** (substantial). Below 0.40 means your guidelines are broken, not your annotators. For reference, BanTH reported Fleiss' κ of 0.71 inter-annotator and 0.75 expert-annotator on binary labelling.

If κ is low, the usual cause is that "hallucination" is under-defined for edge cases — partially correct answers, answers correct but not entailed by the context, answers that hedge. Write explicit rules for each.

### 5.3 What to annotate vs. what to trust

| Data | Annotation level |
|---|---|
| Test set (500) | **100% human, double-annotated, adjudicated** |
| Dev set (500) | 100% human, single annotator + spot check |
| Train set (3,000) | Generator label + human verification on a 20% sample |

If the 20% sample shows > 5% label noise, verify more.

---

## 6. Preprocessing and the three input arms

Build **three parallel preprocessing arms** and run every model through all three. This is cheap and one of them will win.

### Arm A — Raw
Text as-is. Minimal cleaning: strip URLs, collapse whitespace, remove PII.

### Arm B — Normalised (Bengali script)
Back-transliterate Banglish → Bengali script, then apply the csebuetnlp normaliser.

```python
from normalizer import normalize
text_bn = back_transliterate(text_banglish)   # Avro / mBART / indic_transliteration
text_bn = normalize(text_bn)
```

**Important caveat:** the csebuetnlp `normalize()` pipeline is designed for Bengali script. Applying it to Latin-script Banglish is not what it was built for. Use it *after* back-transliteration, and for BanglaBERT/BanglishBERT specifically — those models were pretrained with it, so skipping it degrades results.

### Arm C — Dual
Concatenate raw Banglish and its back-transliteration: `banglish [SEP] bengali_script`. Lets the model use both views.

### Input format variants

Test all three per arm. Format often matters more than architecture:

| Format | Template |
|---|---|
| F1 (plain) | `question + " " + answer` |
| F2 (with context) | `context + " " + question + " " + answer` |
| F3 (NLI-style) | `[CLS] context [SEP] question + answer [SEP]` |

For the has-context split, **F3 usually wins** — framing hallucination as textual entailment lets the model exploit pretrained NLI-like structure. Consider initialising from an XLM-R checkpoint already fine-tuned on XNLI.

---

## 7. Model ladder

### 7.1 Classical baselines (your syllabus models — the lower bound)

Run these first. They are fast, they give you a working pipeline in an afternoon, and their failure modes are informative.

| Model | Config | Also log |
|---|---|---|
| N-gram + LogReg | TF-IDF, word 1–2 grams + char 3–5 grams | **OOV rate** on test |
| N-gram + SVM | Same features, LinearSVC | — |
| Skip-gram + LogReg | gensim Word2Vec sg=1, dim=200, window=5, min_count=2, averaged | **vocab coverage** |
| Skip-gram + XGBoost | Same embeddings | — |
| BiRNN | 1 layer, hidden 256, Skip-gram init | — |
| BiLSTM | 2 layers, hidden 256, dropout 0.3, Skip-gram init | — |
| BiLSTM + Attention | + additive attention over hidden states | — |

Char n-grams matter a lot here — Banglish spelling variance means word-level features fragment badly, and character features partially recover it.

Expect 0.50–0.65 macro-F1 at 4K training examples. That is the correct result, not a failure. These models have no pretraining to fall back on.

### 7.2 Transformer encoders (the real contenders)

**Verified model IDs:**

| Model | HF ID | Architecture note |
|---|---|---|
| BanglishBERT | `csebuetnlp/banglishbert` | **ELECTRA discriminator**, pretrained on Bengali + English. Requires csebuetnlp normaliser. |
| BanglaBERT | `csebuetnlp/banglabert` | ELECTRA discriminator, Bengali only. Expect it to lose on Banglish — that's a finding. |
| BanglaBERT large | `csebuetnlp/banglabert_large` | If VRAM allows |
| MuRIL | `google/muril-base-cased` | BERT, 17 Indian languages **+ their transliterated counterparts** (Wikipedia via IndicTrans + Dakshina). Apache 2.0. |
| MuRIL large | `google/muril-large-cased` | — |
| XLM-R | `xlm-roberta-base` / `-large` | Strong general multilingual baseline |
| mBERT | `bert-base-multilingual-cased` | — |
| IndicBERT v2 | `ai4bharat/IndicBERTv2-MLM-only` | — |
| Mixed-Distil-BERT | See arXiv 2309.10272 | Pretrained on BN-EN-HI code-mixed data |

**Why BanglishBERT and MuRIL are your top candidates:** both saw transliterated/cross-script data during pretraining. MuRIL's model card is explicit that it was trained on transliterated data because that phenomenon is common in the Indian context. Everything else is guessing at romanised Bangla.

**Reality check from a comparable task.** On BanTH (transliterated Bangla, binary hate speech, 37.3K samples), fine-tuned baselines landed:

| Model | Macro-F1 |
|---|---|
| XLM-R | 77.35 |
| CharBERT | 76.61 |
| BanglaBERT | 76.50 |
| MuRIL | 75.29 |
| BanglishBERT | 75.07 |
| mBERT | 74.97 |
| IndicBERT | 74.51 |

Note how tight that spread is — 2.8 points across seven very different models. **Do not expect architecture choice alone to transform your score.** Preprocessing, input format, and further pretraining will move the needle more.

### 7.3 Fine-tuning hyperparameters

Start from BanTH's published setup — it is a validated recipe on closely related data:

```python
LEARNING_RATE = 2e-5
OPTIMIZER     = "AdamW"
MAX_LENGTH    = 512      # drop to 256 if your answers are short — much faster
EPOCHS        = 5
BATCH_SIZE    = 32       # 16 if OOM
EARLY_STOPPING = "validation loss"
WARMUP_RATIO  = 0.1
WEIGHT_DECAY  = 0.01
```

Minimal training loop:

```python
from transformers import (AutoTokenizer, AutoModelForSequenceClassification,
                          TrainingArguments, Trainer, EarlyStoppingCallback)
from sklearn.metrics import f1_score, accuracy_score
import numpy as np

MODEL = "csebuetnlp/banglishbert"
tok = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=2)

def encode(batch):
    return tok(batch["text_a"], batch["text_b"], truncation=True,
               max_length=256, padding="max_length")

def metrics(p):
    preds = np.argmax(p.predictions, axis=1)
    return {"macro_f1": f1_score(p.label_ids, preds, average="macro"),
            "accuracy": accuracy_score(p.label_ids, preds)}

args = TrainingArguments(
    output_dir="out", learning_rate=2e-5, per_device_train_batch_size=32,
    num_train_epochs=5, weight_decay=0.01, warmup_ratio=0.1,
    eval_strategy="epoch", save_strategy="epoch",
    load_best_model_at_end=True, metric_for_best_model="macro_f1",
    seed=42, fp16=True, report_to="none")

trainer = Trainer(model=model, args=args, train_dataset=train_ds,
                  eval_dataset=dev_ds, compute_metrics=metrics,
                  callbacks=[EarlyStoppingCallback(early_stopping_patience=2)])
trainer.train()
```

### 7.4 Cross-validation

At 4K examples, single-split variance is large enough to mislead you about which model is better. Use **5-fold stratified CV on train+dev**, report mean ± std, keep the test set untouched until the end.

Run every configuration with **3 seeds** (42, 1337, 2024) and report the mean. A 1-point difference between two models across a single seed is noise.

---

## 8. Further pretraining (highest-leverage single step)

This is where the biggest gains live, and it is still entirely a BERT-based method.

### 8.1 The recipe (from BanTH, validated on transliterated Bangla)

```
Objective:      Masked Language Modelling
Masking rate:   15%
Learning rate:  1e-5
Batch size:     32
Epochs:         5
Corpus:         unlabeled transliterated Bangla
```

### 8.2 Corpus

**BanglaTLit-PT** — 243K unlabeled transliterated Bangla texts, publicly available:
`kaggle.com/datasets/farihatanjimshifat1/bangla-transliteration-further-pretraining-dataset`

Supplement with:
- **BanglishRev** — 1.74M e-commerce reviews in Bangla/English/Banglish (arXiv 2412.13161)
- **MixSarc** — naturally-occurring Banglish from Facebook (`ajwad-abrar/MixSarc` on HF)
- Your own unlabeled Banglish scrape

### 8.3 Expected gains — calibrate your expectations

From BanTH's binary classification results, comparing base → further-pretrained ("TB-") variants:

| Base model | Base F1 | After FPT | Δ |
|---|---|---|---|
| mBERT | 74.97 | **77.36** | **+2.39** |
| BanglishBERT | 75.07 | 77.12 | +2.05 |
| BanglaBERT | 76.50 | 77.12 | +0.62 |
| XLM-R | 77.35 | 77.04 | **−0.31** |

Two lessons here. First, FPT gains are real but modest — 2 points, not 20. Second, **it does not always help**: XLM-R got slightly worse. The models that gain most are those with the weakest transliterated-text coverage to begin with. So run FPT on mBERT and BanglishBERT first, and treat XLM-R FPT as optional.

### 8.4 Implementation sketch

```python
from transformers import AutoModelForMaskedLM, DataCollatorForLanguageModeling

model = AutoModelForMaskedLM.from_pretrained("bert-base-multilingual-cased")
collator = DataCollatorForLanguageModeling(tokenizer=tok, mlm=True, mlm_probability=0.15)

args = TrainingArguments(output_dir="fpt_out", learning_rate=1e-5,
                         per_device_train_batch_size=32, num_train_epochs=5,
                         save_strategy="epoch", fp16=True, report_to="none")

Trainer(model=model, args=args, train_dataset=unlabeled_ds,
        data_collator=collator).train()
model.save_pretrained("tb-mbert")
```

Then fine-tune `tb-mbert` exactly as in Step 7.3.

> **Note on ELECTRA models.** BanglaBERT and BanglishBERT are ELECTRA *discriminators*, not standard MLM models. `AutoModelForMaskedLM` will not load them cleanly. For those two, either use the paired generator checkpoint (`csebuetnlp/banglishbert_generator`) for the MLM stage, or use the RTD objective. mBERT and XLM-R are the straightforward FPT targets — start there.

---

## 9. The shortcut audit (do this before trusting any score)

### 9.1 The metadata-only probe

Train logistic regression on features that contain **no content words**:

```python
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

def shortcut_features(text):
    toks = text.split()
    n = max(len(toks), 1)
    return [
        len(toks),                                            # token count
        len(text),                                            # char count
        np.mean([len(t) for t in toks]),                      # mean token length
        sum(c.isascii() and c.isalpha() for c in text) / max(len(text),1),  # latin ratio
        text.count(','), text.count('.'), text.count('?'),
        text.count('!'), text.count('"'),
        sum(c.isdigit() for c in text) / max(len(text),1),    # digit ratio
        sum(1 for t in toks if t.isupper()) / n,              # caps ratio
    ]

X = np.array([shortcut_features(t) for t in df.candidate_answer])
clf = LogisticRegression(max_iter=1000).fit(X_train, y_train)
score = f1_score(y_test, clf.predict(X_test), average="macro")
print(f"SHORTCUT PROBE: {score:.3f}")
```

**Decision rule:**

| Probe score | Verdict |
|---|---|
| < 0.55 | Clean. Proceed. |
| 0.55 – 0.60 | Borderline. Investigate which feature carries the signal. |
| > 0.60 | **Artifacted. Regenerate.** |

If it fails, inspect `clf.coef_` to find the leaking feature, fix the generation prompt, regenerate, re-probe.

### 9.2 The three other checks

**Answer-only probe.** Train a full BERT on `candidate_answer` alone, no question, no context. On the has-context split this should be substantially worse than the full-input model. If it isn't, the answer text alone gives away the label — meaning the model isn't checking grounding at all.

**Human-written holdout.** Keep 100–200 test items whose hallucinated answers were written by humans, not generated. If model performance collapses on this slice, your model learned the generator's fingerprint. This is the most honest test you can run, and it becomes a headline result in Phase 2.

**Suspiciously high scores.** Any macro-F1 above 0.95 on has-context should be treated as leakage until proven otherwise. Realistic ceilings are in Section 11.

---

## 10. Score maximisation, in ROI order

Work down this list. Stop when you hit your target.

### 10.1 Input format sweep — highest ROI
Run F1/F2/F3 × Arm A/B/C = 9 configs on your best single model. Often worth 3–6 points. Half a day.

### 10.2 Further pretraining
Section 8. Worth ~2 points on the right base model. One day.

### 10.3 Hard-negative balance
If your model is at 0.90 on easy and 0.55 on hard, adding hard examples to training helps more than any architecture change.

### 10.4 Soft-voting ensemble
Average the predicted probabilities of your top 3 encoders. Reliably worth 1–3 points and it's what the top BLP shared-task teams do.

```python
probs = (p_banglishbert + p_muril + p_xlmr) / 3
preds = (probs[:, 1] > threshold).astype(int)
```

### 10.5 Threshold tuning
Never default to 0.5. Sweep on **dev only**:

```python
best_t = max(np.arange(0.2, 0.81, 0.01),
             key=lambda t: f1_score(y_dev, (p_dev[:,1] > t).astype(int), average="macro"))
```

Apply `best_t` to test once. Worth 1–2 points on imbalanced or miscalibrated models.

### 10.6 Class weights / focal loss
Only if your final class balance drifts from 50/50.

### 10.7 Hyperparameter search — lowest ROI
The BanTH defaults are fine. Don't spend a week here.

---

## 11. LLM reference point

Not a competitor — a **ceiling marker** and reviewer insurance. One afternoon.

Run GPT-4-class, Gemini, and/or Llama zero-shot and few-shot on 300–500 test items. Report as one block in your results table.

**Prompting strategies worth testing** (all validated on transliterated Bangla in BanTH):

| Strategy | Addition to base prompt |
|---|---|
| Non-explanatory | Base only |
| CoT | + "Let's think step by step" |
| Explanation | + "Explain why" |
| **Translation-based** | Prepend: *"Translate the following transliterated text into standard Bangla/English"* then classify |

The translation-based strategy was BanTH's own contribution and it topped their zero-shot results. It is directly applicable to your task and costs nothing to try.

**What to expect:** on BanTH, fine-tuned encoders beat every prompting result by roughly 8 points. Your fine-tuned models should win. If a zero-shot LLM beats your fine-tuned encoder by a wide margin, that's a signal your training data is too small or too noisy.

---

## 12. Evaluation and reporting

### 12.1 Metrics

Primary: **macro-F1** (equal weight to both classes).
Also report: accuracy, per-class precision/recall, AUROC, confusion matrix.

Report **separately** for:
- has-context vs. no-context
- easy vs. hard
- per hallucination type
- human-written holdout vs. generated

### 12.2 Statistical rigour

- 3 seeds, report mean ± std
- **McNemar's test** for pairwise model comparison on the same test set
- Bootstrap 95% CI on the headline number (1000 resamples)

```python
from statsmodels.stats.contingency_tables import mcnemar
# table = [[both_correct, a_only], [b_only, both_wrong]]
print(mcnemar(table, exact=False, correction=True))
```

Without this, you cannot claim model A beats model B when they differ by 1.5 points.

### 12.3 Realistic targets

| Setting | Realistic macro-F1 | Stop-and-check threshold |
|---|---|---|
| Has-context (intrinsic) | **0.80 – 0.90** | > 0.95 → audit for leakage |
| No-context (extrinsic) | **0.60 – 0.75** | > 0.85 → audit |
| Hard subset only | **0.55 – 0.70** | — |
| Classical models @ 4K | 0.50 – 0.65 | Expected |
| Zero-shot LLM | 0.60 – 0.72 | — |

**External anchors:**
- ViHallu (Vietnamese, 3-class, 10K triplets, 111 teams): best system **84.80** macro-F1; encoder-only baseline **32.83**
- BanTH (transliterated Bangla, binary): best **77.36** macro-F1
- MedHallu hard category: best model **0.625** F1
- SHROOM-CAP Bengali zero-shot factuality: ~**0.51** F1

The no-context number will be much lower than has-context. **That is correct, not a bug.** Closed-book factuality without retrieval is genuinely hard. Report it plainly.

### 12.4 Required results tables

**Table 1 — Main results**

| Model | Has-context F1 | No-context F1 | Overall F1 | Acc |
|---|---|---|---|---|
| *Classical* | | | | |
| N-gram + LogReg | | | | |
| Skip-gram + LogReg | | | | |
| BiLSTM | | | | |
| *Pretrained encoders* | | | | |
| mBERT | | | | |
| XLM-R | | | | |
| MuRIL | | | | |
| BanglaBERT | | | | |
| BanglishBERT | | | | |
| *Further pretrained* | | | | |
| TB-mBERT (ours) | | | | |
| TB-BanglishBERT (ours) | | | | |
| *Ensemble* | | | | |
| Soft-vote top-3 | | | | |
| *LLM reference* | | | | |
| GPT zero-shot | | | | |

**Table 2 — Difficulty breakdown** (easy/hard × model)
**Table 3 — Per-hallucination-type F1**
**Table 4 — Preprocessing arm × input format ablation**
**Table 5 — Shortcut audit results**

---

## 13. Timeline

| Week | Deliverable |
|---|---|
| **1** | Licence audit, source selection, schema frozen, 500-item pilot generated, **shortcut probe run on pilot** |
| **2** | Annotation guidelines written, 100-item annotation pilot, κ computed, guidelines revised |
| **3** | Full generation (4,000 items), full annotation, splits created, IAA reported |
| **4** | Classical baselines + first transformer run; pipeline end-to-end |
| **5** | Full model ladder × 3 preprocessing arms × 3 input formats; 5-fold CV |
| **6** | Further pretraining; ensemble; threshold tuning; LLM reference; final audit; results tables |

Six weeks part-time. Weeks 1–3 are the ones people underestimate — data work always takes longer than modelling.

---

## 14. Risk register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| **Artifact leakage inflates scores** | **High** | **Critical** | Shortcut audit (§9) on the pilot, before full generation |
| Low IAA (κ < 0.5) | Medium | High | 100-item pilot, revise guidelines, add edge-case rules |
| Kaggle competition data unusable | Medium | Low | Don't build on it; use as external benchmark only |
| No-context split scores poorly | High | Low | Expected — report honestly, it's a finding |
| Model spread too tight to distinguish | Medium | Medium | 3 seeds + McNemar; report CIs |
| BenHalluEval team publishes first | Medium | Medium | Phase 1 is a course deliverable, not a claim race; Phase 2 positions against them |
| Annotator dropout | Medium | Medium | Recruit 3, need 2 |
| GPU quota exhausted | Low | Medium | Base models only; `max_length=256`; gradient accumulation |

---

## 15. Key references

**Read first:**
- **BanTH** — arXiv 2410.13281. Transliterated Bangla, FPT recipe, full baseline table, annotation guidelines in Appendix B. Your closest methodological template.
- **DSC2025 ViHallu** — arXiv 2601.04711. Vietnamese hallucination shared task; the paper structure you'll eventually write.
- **BenHalluEval** — arXiv 2605.31483. Bengali hallucination benchmark incl. a Bangla–English code-mixed track. Know it in detail.

**Models & resources:**
- BanglaBERT / BanglishBERT — arXiv 2101.00204, `github.com/csebuetnlp/banglabert`
- MuRIL — arXiv 2103.10730, `google/muril-base-cased`
- BEnQA — arXiv 2403.10900, `github.com/sheikhshafayat/BEnQA`
- BanglaTLit — ACL Findings EMNLP 2024 (`2024.findings-emnlp.859`)
- MixSarc — arXiv 2602.21608
- BanglishRev — arXiv 2412.13161
- OffMix-3L — arXiv 2310.18387 (synthetic vs. natural transfer numbers)

**Hallucination detection:**
- MedHallu — `medhallu.github.io` (difficulty stratification, "not sure" category)
- SHROOM-CAP — arXiv 2511.18301 (multilingual, Bengali zero-shot)
- HaluEval, HalluLens (taxonomy)
- Semantic entropy — Farquhar et al., *Nature* (uncertainty baseline for Phase 2)
- `github.com/EdinburghNLP/awesome-hallucination-detection` (living survey)

**Kaggle:**
- অলীকবচন Bengali LLM Hallucination Detection Challenge — `kaggle.com/competitions/bengali-hallucination`
- BanglaTLit-PT FPT corpus — `kaggle.com/datasets/farihatanjimshifat1/bangla-transliteration-further-pretraining-dataset`
- ML Olympiad hallucination detection — public starter notebooks (TF-IDF+LogReg, XGBoost ~0.816, transformer starters)

---

## Appendix A — Day-one checklist

- [ ] Repo created with the Section 1.3 layout
- [ ] `results/experiment_log.csv` created with headers
- [ ] `data/SOURCES.md` created; licence recorded for every source
- [ ] Kaggle competition rules/licence read and recorded
- [ ] Data schema (Section 2) frozen and written to `configs/schema.json`
- [ ] Generation prompt template drafted with length + register constraints
- [ ] Annotation guidelines document started
- [ ] Two annotators recruited and briefed
- [ ] Seeds fixed: 42, 1337, 2024
- [ ] Test set designated and **locked** — no looking at it until Week 6
