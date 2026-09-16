# Phase 1 Implementation Guide -- Bengali

**Scope:** build a working **Bengali** hallucination detector and benchmark the full model
ladder. Bangla script in, Bangla script out.

**Out of scope for Phase 1:** anything Banglish or romanised, transliteration arms, the CMI
degradation study, span-level annotation, and the paper. Those are Phase 2. See
[PRD.md](PRD.md) section 3.2.

**Version:** 2.0 -- rewritten when the project moved from Banglish-first to Bengali-first.
**Last verified:** 23 August 2026

---

## 0. What "success" means in Phase 1

You are done when all six hold:

| # | Criterion | Target |
|---|---|---|
| C1 | Corpus built and annotated, IAA reported | >= 4,000 pairs, Cohen's kappa >= 0.60 |
| C2 | Best model on the has-context **hard subset** | macro-F1 >= 0.80 |
| C3 | Best model on **no-context** | macro-F1 >= 0.60 |
| C4 | **Shortcut audit passes** | metadata probe < 0.60 macro-F1 |
| C5 | Every has-context number reported with its string baseline | 0.812 all / 0.456 hard |
| C6 | Full experiment log exists | every run recorded with seed + config |
| C7 | **No RAG anywhere** | no retrieval at inference; see section 7.0 |

**C4 is not optional.** A high score that fails C4 means the model learned an artifact, not
hallucination detection. **C5 is the one people forget:** a string matcher scores 0.812 on
has-context, so a raw 0.83 is not a result. Read section 3.3 before reporting anything.

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
  "label": 1,
  "hallucination_type": "none | entity | numeric | relational | fabricated | contradiction",
  "difficulty": "easy | hard",
  "generator_model": "…",
  "generation_seed": 42,
  "annotator_1": 0,
  "annotator_2": 0,
  "adjudicated": false,
  "script_condition": "bengali",
  "cmi": 0.0
}
```

### 2.1 Label convention — PROJECT-WIDE, NON-NEGOTIABLE

```
label = 1  ->  CORRECT / FAITHFUL     (no hallucination)
label = 0  ->  INCORRECT / HALLUCINATED
```

Read it as an **is-it-correct?** flag: `1` means yes, `0` means no.

This is the single convention used everywhere — the schema, `data/`, all `src/`
scripts, every results table, and both documents. It matches the native polarity of
the ingested `bn_qa_pool` files, so **no flip is applied anywhere in the pipeline**.

Consequences to keep straight:

- The **positive class (1) is faithful**, not hallucinated. Much hallucination-detection
  literature uses the opposite. When comparing to ViHallu, BanTH, or MedHallu numbers,
  macro-F1 is unaffected (it is symmetric under a global flip), but **per-class
  precision/recall are not** — state which class you are quoting.
- `hallucination_type` is `none` exactly when `label == 1`.
- In a softmax, `probs[:, 1]` is P(correct). Threshold sweeps select on P(correct).
- A detector's "detection" is therefore predicting `0`.

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

## 3. The corpus and how it was built

**Phase 1 does not generate data.** The corpus already exists and is built deterministically.
Read the docstring of `src/build_corpus.py` before changing anything about the data.

```
data/raw/bn_qa_pool/              14 source .jsonl, verbatim, 62,084 records
  |  src/build_bn_pool.py         clean, dedup, map to the frozen schema
data/interim/bn_pool.jsonl        56,480 records (the unfiltered pool)
  |  src/build_corpus.py          pairing + validity filters + splits
data/corpus/bn_v1/corpus.jsonl    4,480 pairs / 8,960 records
data/splits/{train,dev,test}      3,129 / 673 / 678 pairs
  |  src/merge_annotation.py      human annotation (data/annotated/round1/) -> types
```

Both build scripts are deterministic at seed 42 and reproduce byte-identical output. A rebuild
resets `hallucination_type` to `unlabeled`, so rerun `src/merge_annotation.py` after one.

### 3.1 Scope: every subject is included

**Nothing is excluded for being hard.** Law, science, BCS and literature are all in the corpus —
966 pairs, 21.6% of the total — on the same footing as everything else.

**This project is QA only, so fill-in-the-blank items are excluded.** That is a task-type rule,
not a difficulty rule: a cloze item trains span-copying rather than answer checking, and a string
matcher scores 0.929 on those items alone. It also removes `geography`, which was 100% cloze.

An earlier revision filtered on "answerability" and dropped roughly 4,150 pairs. That filter was
removed by decision of the project owner. Difficulty is now **measured, not filtered**: the
`difficulty` field marks which pairs a string matcher can already solve, so hard and easy are
reported separately (section 3.3).

Two consequences you have to actually handle:

1. **Annotators cannot verify everything from memory.** For no-context items they may look things
   up, or mark `unsure`; for has-context items they must never look anything up. See PRD 5.1b.
2. **Report per-subject metrics.** `law`, `science`, `bcs` and `literature` ask for dates,
   article numbers and scientific names that a closed-book model has no way to know. If the model
   collapses on those, that is a result to report, not a bug to hide.

### 3.2 The filters, and what each one is for

Only four things are removed, and none of them is about difficulty:

| Filter | Removes | Why |
|---|---|---|
| Incomplete pairs | groups without both a correct and a wrong answer | cannot form a pair at all |
| **Fill-in-the-blank** | cloze items (শূন্যস্থান পূরণ, `___`) | **QA only.** Teaches span-copying, not answer checking; 0.929 for a string matcher |
| Stranded vowel signs | OCR-damaged text ("বিষয়ের ি") | unreadable, therefore unlabelable |
| Verbatim run >= 6 words | question cut out of its own passage | the question answers itself |
| Repeated question text | duplicate questions | one copy in train and one in test is leakage |

Plus two composition constraints from the PRD, which shape the sample rather than filter for
quality: 60/40 has-context to no-context (D2), and a per-subject cap so no subject dominates a
condition — mathematics alone supplies 59% of the no-context pool and would otherwise swamp it.

**There are no fill-in-the-blank items.** `build_corpus.py` asserts this at the end of every
build, so the corpus cannot silently regain them.

**One filter that deliberately does NOT exist:** pairs whose *wrong* answer is also a verbatim
span of the passage are **kept**. An earlier version dropped them, which was backwards --
removing them leaves only pairs where "appears in the passage" lines up exactly with "is
correct", and that is what drives the string-matcher baseline up. Those pairs are the
shortcut-proof ones.

### 3.3 The two shortcuts this corpus has to fight

**1. Metadata shortcut (the blocking gate).** Surface features of the answer alone -- length,
digit ratio, punctuation -- must not predict the label. Currently **0.534**, gate is < 0.60. PASS

**2. String-matcher shortcut (report it, do not ignore it).** The rule "if the answer appears in
the passage, call it correct" scores **0.812** across all has-context items, because correct
answers in extractive QA are usually verbatim spans and generated wrong answers usually are not.

`difficulty` encodes this per pair. **hard** = the string rule does *not* separate that pair's
two answers, so the model has to actually read the passage.

| Slice | records | String-matcher macro-F1 |
|---|---:|---:|
| All has-context | 5,376 | 0.812 |
| Easy | 3,522 | 0.980 |
| **Hard** | 1,854 | **0.456** |

**Never report a has-context score without these numbers next to it.** A model at 0.83 overall
has beaten a string matcher by one point. The hard subset is where the real result lives.

### 3.4 Rebuilding

```bash
python src/build_bn_pool.py     # only if the raw pool changed
python src/build_corpus.py      # the filter and the splits
python src/audit.py --data data/splits    # MUST pass before using the result
```

---

## 4. The corpus is final

**Do not collect, scrape, generate, or synthesise more data.** The corpus is closed at 4,480
pairs by decision of the project owner. This section exists so the decision is not quietly
reversed later.

If a future phase ever does need more grounded Bengali QA, the options in order were: regenerate
the weak wrong answers with an LLM (about 2,400 has-context pairs have a wrong answer not drawn
from the passage, which is what creates the string shortcut); ingest TyDiQA-GoldP (Bengali),
BanglaRQA, or BEnQA, recording the licence in `data/SOURCES.md` **before** ingesting (D10); or
rewrite the excluded fill-in-the-blank items into real questions — they are cloze today, which is
why they are out, but rewritten they would be valid QA.

**What not to do:** generate distractors with hand-written span-swapping rules. It was tried. It
produced truncated words, sentence fragments, and type mismatches — a *place* offered as the
answer to "what was his father's name?". The existing LLM-written wrong answers are markedly
better.

---

## 5. Annotation

### 5.1 Protocol

- **2 annotators minimum**, both native Bangla speakers. 3 is better.
- Write an explicit guidelines document *before* annotation begins. BanTH's Appendix B is an excellent template to imitate — it gives per-category definitions with worked bilingual examples.
- Run a **pilot on 100 items**, compute agreement, resolve disagreements, revise guidelines, *then* annotate the rest.
- A domain expert (you, or your supervisor) adjudicates disagreements.
- Annotators label: `label` (`1` = correct, `0` = hallucinated), and for **label=0**, `hallucination_type`.

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
| Test set (678 pairs / 1,356 records) | **100% human, double-annotated, adjudicated** |
| Dev set (673 pairs / 1,346 records) | 100% human, single annotator + spot check |
| Train set (3,129 pairs / 6,258 records) | Generator label + human verification on a 20% sample (626 pairs / 1,252 records) |

If the 20% sample shows > 5% label noise, verify more.

**Measured at M3:** the train spot-check annotator disagreed with the corpus label on 40 of the
1,203 rows they could decide (**3.3%**), and dev shows the same rate (43 / 1,314). That is under
the 5% threshold, so by this rule train does not need wider label verification. The disputed rows
are listed in `data/annotated/round1/label_disputes.csv`; labels are never changed by the merge.

---

## 6. Preprocessing and input formats

Phase 1 is Bengali script throughout, so there are **no transliteration arms**. The old
raw / back-transliterated / dual three-arm design belongs to Phase 2, where the input is Banglish
and converting it to Bengali script is a genuine modelling choice. Here it would be a no-op.

What Phase 1 does have is normalisation and input format -- and format usually matters more than
architecture.

### Normalisation -- required for BanglaBERT

BanglaBERT was pretrained on text passed through csebuetnlp's normaliser. Skipping it costs
accuracy for no reason.

```python
from normalizer import normalize
text = normalize(text)        # Bengali script -- correct usage in Phase 1
```

Use the same normalisation at training and at inference, and log which you used. For mBERT,
XLM-R, and MuRIL it is optional -- run it as an ablation rather than assuming.

### Input formats -- sweep all three (M9)

| Format | Template | Notes |
|---|---|---|
| F1 (plain) | `question + " " + answer` | The only option for no-context items |
| F2 (with context) | `context + " " + question + " " + answer` | |
| F3 (NLI-style) | `[CLS] context [SEP] question + answer [SEP]` | Usually best for has-context |

F3 frames the task as textual entailment -- "does this passage entail this answer?" -- which is
close to what pretrained NLI models already do. Starting from an XLM-R checkpoint already
fine-tuned on XNLI is worth trying.

**Truncation matters more than usual here.** Passages in the corpus have a median length of 266 characters
but run up to 3,132. At `max_length=256` a long passage gets cut, and if the supporting sentence is
what got cut, the label is no longer derivable from the input -- you are training on noise.
Truncate the **context**, never the question or the answer.

---

## 7. Model ladder

### 7.0 Technique constraint — RAG is forbidden

**No retrieval-augmented generation and no retrieval of any kind at inference time.** No vector
store, no embedding search over the corpus, no nearest-neighbour lookup of similar training
examples, no fetching external passages. Model input = question + the passage already in the
record (has-context only) + candidate answer.

Retrieval would let a model answer `law` or `science` items by looking the fact up. That measures
retrieval, not hallucination detection, and it destroys the no-context condition entirely.

Using embeddings as **features** is fine and expected — word2vec, skip-gram, and encoder
representations are all just feature extractors here. The prohibition is on *retrieving other
documents or examples at inference*, not on embeddings as such.

Everything non-retrieval is allowed and extensions are welcome: n-gram, skip-gram/word2vec, word
and text embeddings, RNN, LSTM/BiLSTM (+attention), BERT-family encoders, ensembles. The required
ladder below may be extended but not replaced.


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

Char n-grams matter a lot here — Bangla is heavily inflected and compounds freely, so word-level features fragment across surface forms of the same root. Character features partially recover that. They also catch the near-miss wrong answers (পুনর্মিলন vs পুনঃমিলন), which differ by a character or two.

Expect 0.50–0.65 macro-F1 at 4K training examples. That is the correct result, not a failure. These models have no pretraining to fall back on.

### 7.2 Transformer encoders (the real contenders)

**Verified model IDs:**

| Model | HF ID | Architecture note |
|---|---|---|
| **BanglaBERT** | `csebuetnlp/banglabert` | **ELECTRA discriminator**, Bengali only. Your primary Phase 1 model. Requires the csebuetnlp normaliser. |
| BanglishBERT | `csebuetnlp/banglishbert` | ELECTRA discriminator, Bengali + English. Optional here; central in Phase 2. |
| BanglaBERT large | `csebuetnlp/banglabert_large` | If VRAM allows |
| MuRIL | `google/muril-base-cased` | BERT, 17 Indian languages **+ their transliterated counterparts** (Wikipedia via IndicTrans + Dakshina). Apache 2.0. |
| MuRIL large | `google/muril-large-cased` | — |
| XLM-R | `xlm-roberta-base` / `-large` | Strong general multilingual baseline |
| mBERT | `bert-base-multilingual-cased` | — |
| IndicBERT v2 | `ai4bharat/IndicBERTv2-MLM-only` | — |
| Mixed-Distil-BERT | See arXiv 2309.10272 | BN-EN-HI code-mixed. Phase 2 candidate, not Phase 1. |

**Why BanglaBERT and MuRIL are your top Phase 1 candidates:** BanglaBERT is pretrained on Bengali specifically and should lead on native script. MuRIL covers 17 Indian languages and is required by M5. XLM-R is the strong general baseline that often wins anyway — see the spread below.

> MuRIL's transliteration coverage and BanglishBERT's cross-script pretraining are advantages for **Phase 2**, not for Phase 1. Do not pick a Phase 1 model on that basis.

**Reality check from a comparable task.** BanTH is *transliterated* Bangla (so a Phase 2 analogue, not a Phase 1 one), binary hate speech, 37.3K samples. Take the *spread*, not the ordering, as the lesson:

| Model | Macro-F1 |
|---|---|
| XLM-R | 77.35 |
| CharBERT | 76.61 |
| BanglaBERT | 76.50 |
| MuRIL | 75.29 |
| BanglishBERT | 75.07 |
| mBERT | 74.97 |
| IndicBERT | 74.51 |

Note how tight that spread is — 2.8 points across seven very different models. **Do not expect architecture choice alone to transform your score.** Input format and further pretraining will move the needle more. Report ties as ties (McNemar), and expect BanglaBERT to rank higher on Bengali script than it did on that transliterated task.

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

MODEL = "csebuetnlp/banglabert"
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
Corpus:         unlabeled Bengali text (Phase 1) / transliterated Bangla (Phase 2)
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

Two lessons here. First, FPT gains are real but modest — 2 points, not 20. Second, **it does not always help**: XLM-R got slightly worse. The models that gain most are those with the weakest transliterated-text coverage to begin with. In Phase 1 the FPT targets are **mBERT and XLM-R** — they are the standard MLM models in the ladder. BanglaBERT and BanglishBERT are ELECTRA discriminators and cannot be further pretrained with `AutoModelForMaskedLM`; see the note below.

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
Run F1/F2/F3 on your best single model (Phase 1 has no transliteration arms — see section 6). Also sweep normalised vs raw. Often worth 3–6 points. Half a day.

### 10.2 Further pretraining
Section 8. Worth ~2 points on the right base model. One day.

### 10.3 Hard-negative balance
If your model is at 0.90 on easy and 0.55 on hard, adding hard examples to training helps more than any architecture change.

### 10.4 Soft-voting ensemble
Average the predicted probabilities of your top 3 encoders. Reliably worth 1–3 points and it's what the top BLP shared-task teams do.

```python
probs = (p_banglabert + p_muril + p_xlmr) / 3
# column 1 is P(correct) under this project's convention (1 = correct)
preds = (probs[:, 1] > threshold).astype(int)
```

### 10.5 Threshold tuning
Never default to 0.5. Sweep on **dev only**:

```python
best_t = max(np.arange(0.2, 0.81, 0.01),
             key=lambda t: f1_score(y_dev, (p_dev[:,1] > t).astype(int), average="macro"))
# p_dev[:,1] = P(label==1) = P(correct)
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
| IndicBERT v2 | | | | |
| *Further pretrained* | | | | |
| FPT-mBERT (ours) | | | | |
| FPT-XLM-R (ours) | | | | |
| *Ensemble* | | | | |
| Soft-vote top-3 | | | | |
| *LLM reference* | | | | |
| GPT zero-shot | | | | |

**Table 2 — Difficulty breakdown** (easy/hard × model)
**Table 3 — Per-hallucination-type F1**
**Table 4 — Normalisation × input format ablation**
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
- **BenHalluEval** — arXiv 2605.31483. The Bengali hallucination benchmark. Four tasks (generative QA, code-mixed QA, summarisation, reasoning), 12,000 hallucinated candidates generated with GPT-5.4 across twelve hallucination types, sourced from TyDiQA-GoldP, BanglaCHQ-Summ and SOMADHAN. Nine LLMs evaluated under a dual-track protocol; BenHalluScore is a balanced error rate spanning 7.72–55.42%. Native-speaker validation reached κ = 0.911–0.926. **Read it before writing anything up** — it is the work your results will be compared against.
- **অলীকবচন / Bengali LLM Hallucination Detection** — `kaggle.com/competitions/bengali-hallucination`. Possible origin of part of the source pool; licence unresolved (PRD Q1). Read the rules tab.

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
- [ ] Label convention confirmed everywhere: **1 = correct, 0 = hallucinated** (Section 2.1)
- [ ] Generation prompt template drafted with length + register constraints
- [ ] Annotation guidelines document started
- [ ] Two annotators recruited and briefed
- [ ] Seeds fixed: 42, 1337, 2024
- [ ] Test set designated and **locked** — no looking at it until Week 6
