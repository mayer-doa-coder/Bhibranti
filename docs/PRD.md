# Product Requirements Document — Phase 1

**Project:** Hallucination detection for Bengali educational question answering
**Phase 1 scope:** **Bengali only.** Bangla script in, Bangla script out.
**Phase 2 (later):** Banglish — Bangla written in English letters.

> If this document and any other file disagree, **this document wins**, and the other file
> should be fixed.

Last updated: 2026-09-17 — model ladder extended to cover the NLP lab syllabus (M10–M14, §5.3b,
§5.3c, V6); human-flagged pairs excluded from training and scoring (§5.1d);
D8 narrowed to test + dev + train spot-check (§5.1c).
2026-08-23 — the project moved from Banglish-first to Bengali-first (§14).

---

## 1. Executive summary

Large language models answer Bengali questions fluently and confidently, and are sometimes
simply wrong. This project builds a detector that reads a question, an optional passage, and a
candidate answer, and decides whether the answer is **correct** or **hallucinated**.

Phase 1 delivers three things:

1. A filtered, human-checked **Bengali** corpus of QA pairs
2. A benchmarked **model ladder**, from simple word counting up to pretrained transformers, that
   follows the NLP lab syllabus (Labs 1–5) rung by rung (§5.3)
3. A **reproducible pipeline** and a complete record of every experiment

### 1.1 The problem statement, and what Phase 1 answers

**The problem is fixed:** given a Bengali question, an optional passage, and a candidate answer,
decide whether the answer is **correct (1)** or **hallucinated (0)** — without retrieving anything
(§5.3a). Extending the ladder to cover the lab (2026-09-17) changed *how* the problem is studied,
not *what* the problem is.

Phase 1 answers these questions, each with a measured number rather than a claim:

| # | Question | Answered by |
|---|---|---|
| Q-A | How well can a detector separate correct from hallucinated Bengali answers, with and without a passage? | every model, has-context vs no-context (§5.4) |
| Q-B | Does it actually read the passage, or just match strings? | hard subset vs exact and fuzzy string matchers (§5.5, V4, V6) |
| Q-C | How far does each step of the NLP lab syllabus get — counting words, n-gram language models, embeddings, recurrent networks, a Transformer? | M1–M3, M10–M13, in lab order |
| Q-D | How much of the best score comes from **pretraining** rather than architecture? | M13 (Transformer from scratch) vs M4/M5 |
| Q-E | Does standard text preprocessing help or hurt Bengali hallucination detection? | M10 |
| Q-F | Which models use **word order**, and does it matter for relational and negation errors? | M14 |
| Q-G | Which kinds of error (entity, numeric, relational, contradiction, fabricated) and which subjects stay hard? | per-type and per-subject breakdowns (§5.4) |
| Q-H | Is the label trustworthy enough to learn from? | κ = 0.865 on test; 4.5% confirmed noise in the train sample; 180 unusable pairs excluded (§5.1d) |

Banglish is deliberately **not** in Phase 1. It is a harder problem on top of an unsolved one,
and the earlier attempt to start there failed (§14.1). Bengali first, then Banglish.

---

## 2. Background

### 2.1 The problem

Bengali is the sixth most spoken language in the world and remains low-resource for evaluation.
Students in Bangladesh increasingly use LLMs to study for BCS, SSC, and HSC exams. A confident
wrong answer in that setting is worse than no answer, because the student cannot tell the
difference and will memorise it.

### 2.2 What already exists

**BenHalluEval** (arXiv 2605.31483, 2026) is the first dedicated Bengali hallucination benchmark.
It covers four tasks — generative QA, code-mixed QA, summarisation, and reasoning — using 12,000
hallucinated candidates generated with GPT-5.4 across twelve hallucination types, drawn from
TyDiQA-GoldP, BanglaCHQ-Summ, and SOMADHAN. It evaluates nine LLMs and reports a dual-track
score. Three of its findings shape this project:

| Their finding | What we do about it |
|---|---|
| **Dual-track evaluation is necessary.** Measuring only detection rate hides a model that just says "hallucinated" to everything. | We report per-class metrics separately, always. A single averaged number is not accepted (§5.4). |
| **Native-speaker validation reached κ = 0.911–0.926.** | Our κ ≥ 0.60 gate is a floor, not a goal. If we land near 0.6 the guidelines need work. |
| **Code-mixed input behaves differently from native script.** | This is exactly the Phase 2 research question, and the reason Phase 2 is a separate study rather than a variant. |

**Where this project differs.** BenHalluEval evaluates *how well LLMs detect* hallucination by
prompting them. This project *trains a detector* and benchmarks a full ladder of model families
on it, across the full subject range of Bengali exam material, with difficulty measured per item
rather than filtered out (§5.5).

**The অলীকবচন Kaggle competition** (Bengali LLM Hallucination Detection) is a likely origin of
part of the source pool. Its licence terms are still unresolved — see Q1.

### 2.3 Why now

Bengali-capable encoders (BanglaBERT, MuRIL, IndicBERT v2) and Bengali LLMs (TigerLLM, TituLLM,
BanglaLLaMA) are all now available, and there is still no trained Bengali hallucination detector
to compare them on.

---

## 3. Goals and non-goals

### 3.1 Goals (Phase 1)

| ID | Goal | Status |
|---|---|---|
| G1 | A filtered, human-annotated **Bengali** hallucination corpus, ≥ 4,000 QA pairs | ✅ Built (4,480), annotated, adjudicated, merged; 4,300 usable pairs after exclusions (§5.1d) |
| G2 | Cover both **has-context** (grounded) and **no-context** (closed-book) conditions | Done — 60/40 |
| G3 | Benchmark a full model ladder that follows the NLP lab: lexical → n-gram LM → embedding → recurrent → Transformer from scratch → pretrained | Not started |
| G4 | **macro-F1 ≥ 0.80 on the has-context hard subset**, **≥ 0.60 on no-context** | Not started |
| G5 | Pass the **shortcut audit** | Passing (0.534 all records / 0.514 usable) |
| G6 | Reproducible pipeline and complete experiment log | In place |

**G4 changed, and the change matters.** The original target was 0.80 on all has-context items. A
plain string matcher already scores **0.823** there (on the usable data, §5.1d), so that target
measured nothing. The target is now the **hard subset**, where the same string matcher gets
**0.454**. See §5.5.

### 3.2 Non-goals — deferred to Phase 2

- Banglish / romanised Bangla in any form
- Transliteration arms, back-transliteration, code-mixing index (CMI) analysis
- Span-level (which words are wrong) evaluation
- Natural-versus-synthetic transfer study
- Tokenizer fertility analysis
- Fine-tuning generative LLMs as detectors
- Writing the paper

### 3.3 Anti-goals — do not do these

- Do **not** raise a score by weakening the corpus. A high score on a broken corpus is a failed
  deliverable, not a good result.
- Do **not** report a has-context number without the string-matcher baseline next to it.
- Do **not** chase state-of-the-art architectures. The ladder is fixed by course requirement.
- Do **not** use **RAG or any retrieval at inference time** — see §5.3a.
- Do **not** collect, scrape, or generate more data. The corpus is final (§5.1).
- Do **not** start Phase 2 work.

---

## 4. Stakeholders

| Role | Who | Cares about |
|---|---|---|
| Owner / lead | Tawhidul Hasan | Everything below |
| Annotators | Owner + 1 native Bangla speaker | §5.1a, the guidelines, the κ gate |
| Course assessor | Instructor | The full model ladder (M1–M14), use of the lab syllabus (§5.3b), reproducibility |
| Future reader | Phase 2 / paper reviewers | Corpus validity, licence register, honest reporting |

---

## 5. Requirements

### 5.1 Data requirements

**Every subject is included, and nothing is excluded for being hard.** Law, science, BCS and
literature are all in the corpus on the same footing as every other subject — 966 pairs, 21.6%
of the total. Difficulty is measured, never filtered.

**One exception, and it is about task type rather than difficulty: fill-in-the-blank items are
excluded.** This project is question answering only. A cloze item (শূন্যস্থান পূরণ) trains a
model to copy the missing span out of the passage instead of judging whether an answer is
supported, and a plain string matcher scores 0.929 on those items alone. Removing them also
removes `geography`, which was 100% cloze — a consequence of the task-type rule, not a judgement
about geography.

An earlier revision filtered the corpus for "answerability" and removed about 4,150 pairs across
those subjects. **That filter has been removed by decision of the project owner.** What replaced
it is *measurement* rather than exclusion: the `difficulty` field records which pairs a string
matcher can already solve, so hard and easy are reported separately instead of one being
discarded (§5.5).

**The consequence to plan for.** Items in `law`, `science`, `bcs`, and `literature` often ask for
a date, an article number, or a scientific name. An annotator cannot always verify those from
memory, so §5.1b defines what they do instead, and results must be broken down by subject
(§5.4) so any collapse on those subjects is visible rather than averaged away.

**Beyond that, only four things are removed**, none about difficulty: incomplete pairs;
OCR-damaged text; questions containing a 6+ word run copied verbatim out of their own passage;
and duplicate question text (which would leak across splits).

### 5.1b Annotator procedure for items needing outside knowledge

| Condition | May the annotator look it up? |
|---|---|
| **no-context** | **Yes.** Verify, then label, and note that it was verified. |
| **has-context** | **No.** Judge only against the passage — an outside source cannot answer "does this passage support this?" and will cause errors. |

If a no-context item cannot be established even after checking, the annotator writes `unsure`
rather than guessing. `unsure` items are excluded from κ and reported by subject; a high count in
one subject is a finding about the corpus, not annotator failure.

| ID | Requirement | Priority | Status |
|---|---|---|---|
| D1 | ≥ 4,000 QA pairs | Must | ✅ 4,480 |
| D2 | 60/40 has-context / no-context | Must | ✅ 2,688 / 1,792 |
| D3 | 50/50 class balance (`0` hallucinated / `1` correct) | Must | ✅ exact |
| D4 | Test split 100% human-verified, double-annotated | Must | ✅ double-annotated (κ 0.865), disagreements adjudicated, merged into corpus and splits |
| D4a | dev/train may be pre-filled and verified; **test may not** — `hallucination_type` is `none` iff the answer is correct, so pre-filling it would reveal the label | Must | ✅ enforced in `build_annotation_sheets.py` |
| D5 | Cohen's κ ≥ 0.60 on the binary label | Must | ✅ 0.717 (M2, 100-item pilot) and ✅ 0.865 (full 1,356-item test split) |
| D6 | No subject excluded for difficulty; QA-only (no cloze) | Must | ✅ 13 subjects |
| D7 | Pairs never split across train/dev/test | Must | ✅ verified |
| D8 | Hallucination type labelled for every `label == 0` item **in test, dev, and the 20% train spot-check** (scope narrowed — see §5.1c) | Must | ✅ met 2026-09-17: of 1,977 in-scope wrong answers, 1,867 typed (1,724 by annotator agreement, 143 adjudicated), 110 excluded by the adjudicator (89 dispute, 21 skip). Merged into corpus + splits. The other 2,503 train wrong answers are out of scope by design |
| D9 | Easy/hard difficulty labelled | Must | ✅ by construction (§5.5) |
| D10 | Licence recorded for every source before ingestion | Must | ⚠ 2 open questions |
| D11 | `script_condition` and `cmi` populated (Phase 2 payload — never evaluated on) | Should | ✅ |
| D12 | `error_span` captured where available | Could | ⬜ |
| D13 | No PII in released data | Must | ⬜ scan pending |
| D14 | Pairs human review found unusable are excluded from training **and** scoring (§5.1d) | Must | ✅ 180 pairs flagged `excluded`; `src/splits.py` filters them |

### 5.1c Decision — D8 covers test, dev and the train spot-check only (2026-09-17)

**Decision by the project owner.** D8 originally required a hallucination type on every wrong
answer in the corpus (4,480). It now requires one on every wrong answer a human checked: all of
test (678), all of dev (673), and the 20% train spot-check (626) — **1,977** in total.

**Why this is sound, not a shortcut:**

| Reason | Detail |
|---|---|
| Types are reporting metadata | They exist so results can be broken down by error kind (§5.4). No model is trained on or given the type; it would reveal the label. |
| Type breakdowns are reported on dev and test only | Nobody reports a result on train, so a type on a train record is never read by any table. |
| The labels themselves are unaffected | D8 is about the *type*. The *binary label* on all train records stays, and the spot-check measured its confirmed noise at **4.5%** after adjudication (56 of 1,252 records), under the 5% threshold that would trigger wider verification (guide §5.3). |
| The saving is large and the cost is zero | ~2,500 annotations that would change no reported number. |

**What this does *not* change:** D4 (test fully double-annotated) and D5 (κ ≥ 0.60) are untouched,
and so is every model input.

**How it is recorded in the data.** Out-of-scope train records keep
`hallucination_type = unlabeled` with `type_source = outside_train_sample`, so "not required" can
never be confused with "forgotten". Items the adjudicator marks `skip` or `dispute` are reported
exclusions (`adjudicator_skip` / `adjudicator_dispute`). `src/audit.py` flags `unlabeled` only
where D8 still requires a type.

**Reporting obligation.** Any per-type table must say it covers dev/test, and the report must state
that train types were sampled (20%) rather than exhaustive.

### 5.1d Decision — human-flagged pairs are left out of training and scoring (2026-09-17)

**Decision by the project owner (resolves Q5).** A pair is excluded when human review showed its
labels cannot be trusted:

| Reason (`exclusion_reason`) | Meaning | train | dev | test |
|---|---|---:|---:|---:|
| `wrong_answer_confirmed_correct` | the adjudicator marked the stored-wrong answer `dispute` | 28 | 19 | 42 |
| `correct_answer_judged_wrong` | every annotator who saw the stored-correct answer said `wrong` | 28 | 34 | 17 |
| `broken_item` | the adjudicator marked it `skip` | 8 | 5 | 8 |
| **pairs excluded** (a pair can have two reasons) | | **62** | **54** | **64** |
| **usable pairs** | | **3,067** | **619** | **614** |

**Why.** Scoring a model against a label humans confirmed wrong penalises it for being right, and
training on one teaches it the error. The rule was fixed **before any model was trained or
scored**, and uses only human judgements, so it cannot be tuned to favour a result.

**How it is applied.**

- The whole pair is dropped, never one record: that keeps the 50/50 balance and the pairing the
  splits rely on.
- Nothing is deleted. Both records stay in the files with `excluded = true` and an
  `exclusion_reason`, so the decision is auditable and reversible.
- Every training and evaluation script loads data through `src/splits.py`, which removes excluded
  pairs by default. There is one place that applies the rule.
- `src/audit.py` runs the gate on all records *and* on the usable records; both must pass (0.534
  and 0.514).

**Consequences to report.**

- The string-matcher baselines are re-measured on the usable data and become the reference
  numbers (§5.5): **0.823** all has-context, **0.987** easy, **0.454** hard.
- Only human-checked pairs can be flagged. The unchecked 80% of train keeps its measured ~4.5%
  label noise, which is under the 5% rule (guide §5.3).
- The labels themselves are never edited (§5.1a). The excluded pairs are a corpus finding: the
  report must state how many were excluded, for which reasons, and in which split.

### 5.1a Label convention — project-wide, non-negotiable

> **`1` = correct / faithful. `0` = incorrect / hallucinated.**

The field answers *"is this answer correct?"*. The **positive class is faithful**, which is the
reverse of most hallucination papers — there, positive usually means hallucinated.

- `probs[:, 1]` is P(correct)
- The source files already use this convention. **No flip is applied anywhere.**
- Never flip it in a loader, a metric, or a table. If a per-class number looks swapped, the
  reading is wrong, not the data.

### 5.2 Taxonomy requirement

| Condition | Types |
|---|---|
| Intrinsic (has-context) | `entity`, `numeric`, `relational`, `contradiction` |
| Extrinsic (no-context) | `fabricated`, `overclaim` |
| Correct answers | `none` |
| Not yet annotated, or out of D8 scope (train outside the spot-check, §5.1c) | `unlabeled` |

### 5.3 Model requirements

Fixed by course requirement. **Adding models is fine; replacing a required one is not.**

The ladder follows the NLP lab syllabus (`NLP Lab/`, Labs 1–5) rung by rung, then goes beyond it
with pretrained encoders. §5.3b maps every lab topic to where it is used, or why it is not.

| ID | Requirement | Lab |
|---|---|---|
| M1 | **Sparse n-gram baselines.** Bag of Words (counts) and TF-IDF, word 1–2 and character 3–5 grams, each feeding **Naive Bayes** (Laplace, α = 1), Logistic Regression and linear SVM | 2, 3 |
| M2 | **Skip-gram embeddings** trained on the train split only. Document vector by **mean** and by **TF-IDF-weighted mean**, each feeding Logistic Regression and XGBoost | 3 |
| M3 | **Recurrent models in PyTorch:** vanilla (one-direction) RNN, BiRNN, stacked BiLSTM, and BiLSTM + **dot-product attention** (`torch.bmm`). Skip-gram initialisation through `nn.Embedding.from_pretrained` | 4 |
| M4 | ≥ 5 pretrained transformer encoders fine-tuned | beyond |
| M5 | **BanglaBERT and MuRIL** specifically included | beyond |
| M6 | Further pretraining on ≥ 1 encoder (**mBERT or XLM-R only** — see below) | beyond |
| M7 | Soft-voting ensemble of the top 3 | beyond |
| M8 | Zero-shot LLM reference point (≥ 1 model, ≥ 300 items) | beyond |
| M9 | 3 input formats swept on the best model | beyond |
| M10 | **Preprocessing ablations** on the M1–M3 models: Bengali regex cleaning and tokenization (default), stop-word removal and rule-based Bengali stemming (variants). Negation words are never removed (§5.3c) | 1 |
| M11 | **Character n-gram language model** (chain rule, Markov assumption, MLE with Laplace smoothing), used two ways: (a) a passage-conditioned score of the answer, (b) a class-conditional classifier (one LM per label, trained on train answers) | 2, 3 |
| M12 | **Similarity features within the record:** normalised Levenshtein edit distance and Skip-gram cosine similarity between the answer and its own passage / question, fed to Logistic Regression. Also defines the **fuzzy string-matcher baseline** (V6) | 1, 3 |
| M13 | **Transformer encoder trained from scratch** in PyTorch: sine/cosine positional encoding, multi-head self-attention with a padding mask, masked mean pooling, single-logit output. Measures what pretraining adds by comparison with M4 | 5 |
| M14 | **Word-order diagnostic:** re-score dev with the words of each input shuffled; report the drop for every M1–M4 / M13 model, plus per-type results (relational errors are where order matters) | 3 |

Deterministic models (Naive Bayes, the n-gram LM, rule baselines) run once — a seed cannot change
them. Every trained neural or stochastic model runs on all three seeds.

> **BanglaBERT is an ELECTRA discriminator, not a masked LM.** `AutoModelForMaskedLM` will not
> work on it. Further pretraining targets mBERT and XLM-R. BanglishBERT is optional in Phase 1
> and becomes central in Phase 2.

### 5.3b Lab coverage — every lab topic, used or excluded (decided 2026-09-17)

The NLP lab (`NLP Lab/`, 5 guides + 3 notebooks) teaches 33 topics. Each was checked against five
questions: does it help decide correct vs hallucinated; does it work on Bengali; does it keep the
evidence of an error (digits, negation, small character changes); does it respect the constraints
(no RAG, no new data, test once); and does it produce a finding worth reporting.

**Used (26 topics):**

| Lab | Topic | Where in this project |
|---|---|---|
| 1 | Regex text cleaning | M10 — Bengali-aware cleaning (citation marks `[1]`, whitespace, stray quotes). Never removes digits or Bengali letters |
| 1 | Tokenization | M10 — Bengali word tokenizer (spaces, দাঁড়ি `।`, punctuation) for M1–M3, M11–M14 |
| 1 | Stop-word removal | M10 — ablation only, negation-preserving list |
| 1 | Stemming | M10 — ablation only, rule-based Bengali suffix stripper written in the style of Porter's rule steps; never strips negation |
| 1 | Edit distance (Levenshtein) | M12 features + V6 fuzzy string-matcher baseline |
| 2 | Bag of Words | M1 |
| 2 | TF-IDF | M1 |
| 2 | N-grams as features | M1 |
| 2 | N-gram language model (chain rule, Markov, MLE) | M11 |
| 3 | Word2Vec Skip-gram | M2 |
| 3 | Cosine similarity | M12 features; offline sanity check of the Bengali word vectors |
| 3 | Naive Bayes + Laplace smoothing | M1; smoothing also in M11 |
| 3 | Logistic Regression | M1, M2, M12 |
| 3 | Mean embedding | M2 |
| 3 | TF-IDF-weighted embedding | M2 |
| 3 | Word-order limitation of bag-of-words models | M14 |
| 4 | PyTorch training loop | M3, M13 |
| 4 | Pretrained embeddings in `nn.Embedding` (PAD/UNK) | M3 |
| 4 | Vanilla RNN | M3 |
| 4 | BiRNN | M3 |
| 4 | Stacked BiLSTM | M3 |
| 4 | Attention (`torch.bmm`) | M3 |
| 4 | Padding / loss masking | M3 (packed sequences), M13 (padding mask) |
| 5 | Transformer encoder from scratch | M13 |
| 5 | Positional encoding | M13 |
| 5 | Multi-head self-attention, padding mask, mean pooling | M13 |

**Excluded (7 topics), with the reason:**

| Lab | Topic | Why it is not used |
|---|---|---|
| 1 | Lemmatization | Needs a dictionary and part-of-speech tags; WordNet is English and no dependable Bengali lemmatizer exists. The stemming ablation tests the same idea |
| 1 | Spelling correction | It would "correct" exactly the errors this project detects (সরখার → সরকার). Typo-only pairs are already excluded (§5.1d) |
| 2 | Shannon's guessing game / n-gram text generation | This project detects; it does not generate (§7.2). Generated text would also be new data (§5.1) |
| 3 | Word analogies (king − man + woman) | A demonstration, not a detector. The useful part — checking that the vectors are sensible — is kept as a cosine-similarity sanity check |
| 4 | LSTM language model + temperature sampling | Generation again. Language-model *scoring* is already covered by M11 |
| 4 | Sequence labelling (POS tagging) | Needs per-word labels this corpus does not have. Marking *which words* are wrong is span detection — Phase 2 |
| 4 | Seq2Seq translation + teacher forcing | There is no translation task. Banglish is Phase 2, and §14.1 requires a real transliteration tool |

**Not adopted by decision (2026-09-17):** pretrained external Bengali word vectors (fastText).
M2/M3 use Skip-gram vectors trained on the train split, which is what Lab 3 and Lab 4 teach.

### 5.3c Preprocessing rules for Bengali

The lab code is written for English. Copying it onto Bengali silently destroys the data:

| Lab code | What it would do here | Rule |
|---|---|---|
| `re.sub(r'[^A-Za-z\s]', '', text)` | deletes **every Bengali letter and every digit** | Bengali-aware patterns only; digits are always kept (475 `numeric` errors live in them) |
| NLTK `word_tokenize`, `stopwords('english')`, `PorterStemmer`, `WordNetLemmatizer` | English-only; wrong or no-op on Bengali | never used on Bengali text |
| Stop-word removal | Bengali lists contain **না, নয়, নি, নেই, নাই** — removing them turns "বিভক্ত নয়" into "বিভক্ত" and erases every negation-flip `contradiction` error (212 typed) | negation words are always kept; removal is an ablation, never the default |
| Stemming | a naive suffix rule strips the negation in "করেননি" | negation suffixes are never stripped; stemming is an ablation, never the default |

### 5.3a Technique constraint — RAG is forbidden

**No retrieval-augmented generation, and no retrieval of any kind at inference time.** That
rules out vector stores, embedding search over the corpus, nearest-neighbour lookup of similar
training examples, and fetching any external passage. The model's input is exactly the question,
the passage already stored in the record (has-context only), and the candidate answer.

Why it matters here: retrieval would let a model answer `law` or `science` items by looking the
fact up, which measures retrieval quality rather than hallucination detection, and would make
the no-context condition meaningless.

**Everything else is permitted**, and adding techniques beyond the required ladder is encouraged:
n-gram models, skip-gram / word2vec, word and text embeddings, RNN, LSTM/BiLSTM with or without
attention, BERT-family encoders, ensembles, and any other non-retrieval method. The required
ladder in §5.3 may be *extended* but not *replaced*.

**Similarity is not retrieval — as long as it stays inside the record.** M12's edit distance and
cosine similarity compare an answer with *its own* question and passage, and M11's passage LM is
built from *that record's* passage. That is allowed. Comparing an answer with *other* records,
or building any lookup over the training set for use at inference, is retrieval and is forbidden.
M11's class-conditional LMs are trained models (like Naive Bayes), not a lookup.

**Other places retrieval could sneak in — all forbidden:**

| Situation | Allowed | Forbidden |
|---|---|---|
| LLM reference (M8) prompts | zero-shot; or few-shot with **one fixed set** of examples, drawn once from train (seed 42) and identical for every test item | picking the few-shot examples *per item* by similarity to it — that is retrieval |
| Further pretraining (M6) | unlabeled Bengali text used to *train* the encoder, before any evaluation | fetching that text, or anything else, at prediction time |
| Ensemble (M7) | averaging the probabilities of trained models | weighting models by how they did on "similar" training items |
| Annotation helpers (`src/llm_annotate.py`) | not part of any detector; never run | — |

**Checked 2026-09-17:** no retrieval code or dependency exists in `src/`, `notebooks/`, `configs/`
or `requirements.txt` (no vector store, search library, nearest-neighbour index or web fetch).
The only network calls are in `src/llm_annotate.py`, which labels data and was never run.

### 5.4 Evaluation requirements

- Primary metric: **macro-F1**. Also report accuracy, per-class P/R/F1, and AUC.
- **Always break results down by:** has-context vs no-context, **easy vs hard**, **by subject**,
  and hallucination type. The type breakdown is computed on **dev and test only** — train types
  exist only for the 20% spot-check (§5.1c).
- Subject breakdown is required because `law`, `science`, `bcs` and `literature` need outside
  knowledge a closed-book model does not have. A collapse there is a finding, not a defect.
- Never report a single averaged number on its own. It hides the only differences that matter.
- Three seeds (42, 1337, 2024), report mean ± std.
- McNemar's test and bootstrap CI before claiming one model beats another.
- **Answer-only models** (M11b class-conditional LM, and any model given only the answer) are
  reported next to the V3 answer-only probe. If they score far above it, they have learned the
  answer generator's style, and that is the finding.
- **Word-order diagnostic (M14)** and **preprocessing ablations (M10)** are run and reported on
  **dev only**. They inform choices; they never touch test.

### 5.5 Validity requirements — the gate

| ID | Requirement | Status |
|---|---|---|
| V1 | Metadata-only shortcut probe scores **< 0.60** macro-F1 — on all records **and** on the usable records | ✅ 0.534 / 0.514 |
| V2 | Audit rerun after any change to corpus construction | ✅ enforced |
| V3 | Answer-only probe substantially below the full-input model | ⬜ needs a model |
| V4 | Every has-context score reported next to the string-matcher baseline | ⬜ enforce at M5 |
| V5 | Any macro-F1 > 0.95 triggers a leakage investigation | ⬜ |
| V6 | The **fuzzy string-matcher** baseline (M12: "correct if the answer *nearly* appears in the passage", edit-distance threshold chosen on train) is reported beside the exact string matcher on every has-context table | ⬜ measure at M4 |

**V1 is a blocking release gate.** Weakening the threshold, the feature set, or the split to make
it pass is the single most damaging thing anyone can do to this project.

**The string-matcher baseline (V4).** The rule *"if the answer text appears in the passage,
call it correct"* scores:

| Slice | Usable records (§5.1d) — **the reference** | All records (before exclusion) |
|---|---:|---:|
| All has-context | **0.823** (n = 5,092) | 0.812 (n = 5,376) |
| Easy subset | 0.987 (n = 3,398) | 0.980 (n = 3,522) |
| **Hard subset** | **0.454** (n = 1,694) | 0.456 (n = 1,854) |

Models are trained and scored on usable records only, so the left column is what every score is
compared with. When a score is reported on dev or test, the baseline is recomputed on that same
split, so both numbers always describe identical records.

**Why a second, fuzzy baseline (V6).** The exact rule is defeated by any one-character change, so it
looks worse on the hard subset (0.454) than a slightly smarter rule would. If a learned model beats
0.454 but not the fuzzy matcher, it has learned approximate string matching, not grounding. Both
numbers are reported so the bar is honest.

This is a property of the source data: correct answers in extractive QA tend to be verbatim spans
of the passage, and the generated wrong answers usually are not. `difficulty` encodes it —
**hard** means the string rule does *not* separate the two answers of that pair, so the model has
to actually read. A has-context score is meaningless without the hard-subset number beside it.

### 5.6 Reproducibility requirements

| ID | Requirement | Status |
|---|---|---|
| R1 | `results/experiment_log.csv` records every run, including failures | ✅ |
| R2 | All seeds fixed and logged | ✅ 42, 1337, 2024 |
| R3 | Any generation prompts version-controlled | ⬜ |
| R4 | Annotation guidelines written *before* annotation begins | ✅ |
| R5 | Environment pinned (`requirements.txt`) | ✅ |
| R6 | Splits deterministic and script-generated | ✅ byte-identical on rerun |

---

## 6. Success metrics

### 6.1 Primary

| Metric | Target | Baseline to beat |
|---|---|---|
| macro-F1, has-context **hard subset** | **≥ 0.80** | 0.454 (string matcher, usable data) |
| macro-F1, has-context overall | report only | 0.823 (string matcher, usable data) |
| macro-F1, has-context (all and hard) | report only | fuzzy string matcher (V6) — measured at M4 |
| macro-F1, no-context | **≥ 0.60** | 0.333 (majority class) |
| Metadata shortcut probe | **< 0.60** | — |
| Cohen's κ | **≥ 0.60** | — |

### 6.2 Secondary

- Every model family on the ladder benchmarked and logged
- Clear, honest ordering across the ladder — including reporting ties as ties
- Per-type breakdown on dev and test (annotation complete, §5.1c)

### 6.3 External calibration

BenHalluEval's BenHalluScore ranges 7.72%–55.42% across models and tasks (lower is better; it is
a balanced error rate). Our numbers are not directly comparable — different task, different data,
trained rather than prompted — so do not claim a head-to-head win.

---

## 7. Scope boundaries

### 7.1 In scope

- Bengali script only, throughout
- Educational domain (BCS / SSC / HSC style), plus Bangla grammar and vocabulary
- Binary hallucination detection at the answer level
- The fixed model ladder (§5.3)
- Both has-context and no-context conditions

### 7.2 Out of scope

- Anything Banglish or romanised (Phase 2)
- Span-level detection, CMI analysis, tokenizer fertility (Phase 2)
- Generation / correction of answers — this project detects, it does not fix. That also rules out
  the lab's text-generation, spelling-correction and Seq2Seq translation exercises (§5.3b)
- Sequence labelling of any kind (per-word tags) — needs labels this corpus does not have
- Languages other than Bengali
- Real-time or production serving

---

## 8. Milestones

| ID | Milestone | Gate to pass | Status |
|---|---|---|---|
| M0 | Foundations: repo, schema, seeds, source pool ingested and audited | Audit written | ✅ |
| M1 | Corpus built, filtered for validity (QA only), split by pair | **V1 metadata probe < 0.60** | ✅ 0.534 |
| M2 | Annotation protocol proven | **Cohen's κ ≥ 0.60** on 100 blind items | ✅ **0.717** |
| M3 | Corpus annotated (test + dev + train spot-check), splits locked | D4, D8 (§5.1c) met | ✅ κ 0.865; 253 rows adjudicated; merged 2026-09-17; strict audit passes (0.534 / 0.514 usable). Q5 decided (§5.1d) |
| M4 | Model ladder trained (M1–M5, M10–M13) | All families logged; V6 measured | ⬜ **next** |
| M5 | Sweeps, further pretraining, ensemble, LLM reference, word-order diagnostic (M6–M9, M14) | V3, V4, V6 enforced | ⬜ |
| M6 | Final evaluation on `test.jsonl` — **once** | Targets in §6.1 | ⬜ |

---

## 9. Deliverables

| ID | Deliverable |
|---|---|
| DL1 | The annotated Bengali corpus |
| DL2 | Annotation guidelines |
| DL3 | Inter-annotator agreement report — ✅ `data/annotated/agreement_test_v1/IAA_REPORT.md` |
| DL4 | Source and licence register |
| DL5 | Trained model checkpoints |
| DL6 | Results tables 1–8 (guide §12.4) |
| DL7 | Experiment log |
| DL8 | Validity audit report |
| DL9 | Reproducible codebase |
| DL10 | Phase 1 technical report |

---

## 10. Risks

| ID | Risk | Mitigation |
|---|---|---|
| RK1 | A shortcut inflates scores and the corpus is invalid | V1 gate, run on every rebuild |
| RK2 | κ below 0.60 | Fix the guidelines, not the annotators; rerun the same 100 items |
| RK3 | Second annotator drops out | Keep the test set small enough for one person to double-label |
| RK4 | Kaggle competition licence forbids redistribution | Q1; keep those items separable |
| RK5 | Model spread too tight to distinguish | 3 seeds + McNemar; report ties honestly |
| RK6 | No-context underperforms | Expected. It is a harder condition, not a bug |
| RK8 | GPU quota exhausted | Base-size encoders, `max_length=256`, fp16 |
| RK9 | Scope creep into Phase 2 | §3.2; flag and stop |
| RK10 | Test set contaminated by repeated evaluation | Dev for everything; test exactly once at M6 |
| RK11 | **Lookup-heavy subjects (law/science/bcs) depress annotation quality** | Accepted deliberately — they are in scope. §5.1b lets annotators verify or mark `unsure`; results are broken down by subject so any effect is visible |

---

## 11. Dependencies

**External:** Bengali Wikipedia (CC BY-SA 4.0), BCS question banks (licence unresolved),
`csebuetnlp/banglabert` + `normalizer`, `google/muril-base-cased`, `xlm-roberta-base`,
`bert-base-multilingual-cased`, Kaggle or Colab GPU.

**Internal:** the second annotator; the owner's own read-through of the corpus.

---

## 12. Open questions

| ID | Question | Blocks |
|---|---|---|
| Q1 | What licence covers the অলীকবচন Kaggle competition data, if the pool draws on it? | DL1 release |
| Q2 | Did the BCS questions come from an official PSC source or a commercial compilation? | Release of those items |
| Q3 | What licence does the released corpus carry? (Wikipedia share-alike likely forces CC BY-SA 4.0) | DL1, DL4 |
| Q4 | Which Claude model and prompts produced the original QA pairs? | R3 |
| Q5 | ~~How are the 180 human-flagged pairs treated?~~ **Decided 2026-09-17: excluded from training and scoring** — §5.1d | — |

---

## 13. Definition of done

Phase 1 is done when all of the following hold:

1. `data/splits/{train,dev,test}.jsonl` are built, locked, and annotated
2. V1 passes and the audit report is committed
3. Every model family M1–M14 has been run (3 seeds where randomness exists) and logged
4. Has-context **hard subset** ≥ 0.80 and no-context ≥ 0.60, or a written explanation of why not
5. Every has-context number in every table appears next to the exact and fuzzy string-matcher
   baselines
6. `test.jsonl` has been evaluated exactly once
7. The licence register has no open questions
8. A fresh clone reproduces the corpus byte-for-byte

---

## 14. Transition to Phase 2

Phase 2 is the Banglish study: build a code-mixed corpus, run the same pipeline, compare, and
measure how code-mixing degrades detection.

### 14.1 What went wrong the first time — read this before starting Phase 2

The project originally began with Banglish. A **rule-based transliteration** (a hand-written
grapheme map plus a small word dictionary) converted Bengali to romanised Bangla. It produced
output that a **native Bangla speaker could not read**, and the 500-pair pilot built on it had to
be discarded.

**The lesson: do not transliterate with hand-written character rules.** Phase 2 must use a real
transliteration model or library, and every batch must be read by a native speaker *before* any
annotation is built on top of it.

### 14.2 What Phase 1 hands over

- A working, audited pipeline that only needs different input data
- A frozen schema that already carries `script_condition`, `cmi`, and `error_span`
- Annotation guidelines proven to reach κ ≥ 0.60
- Bengali baseline numbers to measure code-mixed degradation against

---

## Appendix — terminology

| Term | Meaning |
|---|---|
| **has-context** | A passage is given; the answer must be supported by it |
| **no-context** | No passage; closed-book |
| **hard** | The string shortcut does not separate this pair's two answers |
| **pair** | One question with one correct and one hallucinated answer |
| **intrinsic** | Contradicts the given passage |
| **extrinsic** | Not checkable against any given passage |
| **shortcut** | A surface feature that predicts the label without solving the task |
| **Banglish** | Bangla written in English letters — **Phase 2 only** |
