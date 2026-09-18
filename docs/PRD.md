# Product Requirements Document — Phase 1

**Project:** Detect hallucinated answers in Bengali educational Q&A.
**Phase 1:** Bengali script only.
**Phase 2 (later, not started):** Banglish (Bengali written in English letters).

> **Rule:** if this document disagrees with any other file, this document wins. Fix the other file.

Last updated: **17 September 2026**.

---

## 📍 STATUS AT A GLANCE

Read this table first. It tells you what is done and what is next, without digging through the rest of the document.

| Phase | What it covers | Status |
|---|---|---|
| Data collection | Getting the raw Bengali Q&A pairs | ✅ Done — closed, no more data will be added |
| Corpus building | Cleaning, pairing, splitting into train/dev/test | ✅ Done — 4,480 pairs |
| Shortcut gate | Checking the data has no cheap "trick" a model could exploit | ✅ Passing |
| Annotation pilot | Testing that two humans agree on labels | ✅ Done — agreement score 0.717 |
| Full annotation | Humans checking every test/dev item and a slice of train | ✅ Done — agreement score 0.865 on test |
| Merge + cleanup | Writing human labels into the corpus, removing 180 bad pairs | ✅ Done |
| **Model ladder** | **Training and testing all the required models** | 🔶 **In progress** — step 1 of 11 done: Bengali text tools (`src/text_bn.py`). Next: `src/features.py` |
| Final evaluation | Scoring the locked test set exactly once | ⬜ Not started |

**In one sentence: the dataset is finished and verified. Model building has started (text tools done); the models themselves come next.**

**Where to find things in this document:**

| I want to know... | Go to |
|---|---|
| What exactly is this project trying to solve? | §1 |
| What data do we have, and what are the rules about it? | §5.1 |
| What decisions were made about messy/borderline data? | §5.1c, §5.1d |
| What models must I build, and in what order? | §5.3 |
| Which lab topics are used, and which are skipped? | §5.3b |
| What counts as "done" for this whole project? | §13 |

---

## 1. What problem are we solving?

A model reads a **question**, sometimes a **passage**, and a **candidate answer**. It must decide:

> Is this answer **correct** (label `1`) or **hallucinated** (label `0`)?

The model may **not** search the internet or look anything up. It must decide using only what it was trained on and what is directly given to it (§5.3a).

Phase 1 delivers three things:

1. A cleaned, human-checked **Bengali** dataset of question–answer pairs.
2. A **ladder of models** — starting from simple word-counting, ending at pretrained transformers — that follows the course's NLP lab step by step (§5.3).
3. A **repeatable pipeline**: every script can be re-run and gives the same result, and every experiment is logged.

**Why this matters:** AI models answer Bengali questions confidently, but are sometimes wrong. A wrong answer that *sounds* right is more dangerous than no answer, because a student cannot tell the difference.

**Banglish is not part of Phase 1.** It is a separate, harder problem, tackled later (Phase 2). An earlier attempt to start with Banglish failed — see §14.1.

### 1.1 The eight questions this project must answer

Each of these gets a real number, not a guess.

| # | Question | Where the answer comes from |
|---|---|---|
| Q-A | How well can a model spot a hallucinated Bengali answer, with or without a passage? | Every model's score, split by has-context vs no-context (§5.4) |
| Q-B | Does the model actually *read* the passage, or does it just check if the answer's words appear in it? | Score on the "hard" subset, compared to two string-matching baselines (§5.5) |
| Q-C | How far can each NLP lab technique go — counting words, n-gram models, embeddings, RNNs, a Transformer? | Models M1, M2, M3, M10–M13, tested in that order |
| Q-D | How much of a pretrained model's score comes from its pretraining, not its design? | Compare a Transformer built from scratch (M13) against pretrained ones (M4/M5) |
| Q-E | Does standard text cleanup (like removing common words) help or hurt on Bengali? | M10 |
| Q-F | Do models actually use word order? Does that matter for word-swap and negation errors? | M14 |
| Q-G | Which error types and school subjects are hardest? | Score broken down by type and subject (§5.4) |
| Q-H | Can we trust the labels enough to train on them? | Yes — agreement score 0.865 on test; only 4.5% noise found in a checked sample of train; 180 bad pairs already removed (§5.1d) |

**Note:** adding more model types to study (in September 2026) changed *how* we study the problem. It did **not** change the problem itself.

---

## 2. Background

### 2.1 Why this matters

Bengali is the sixth most-spoken language in the world, but there is little data to test AI systems on it. Students in Bangladesh use AI to study for exams (BCS, SSC, HSC). A confidently wrong answer is worse than no answer, because the student can't tell it's wrong and may memorise it.

### 2.2 What already exists

**BenHalluEval** (2026) is the first Bengali hallucination benchmark. It tests nine AI models on four tasks, using 12,000 wrong answers written by GPT-5.4.

Three things we learned from it, and what we did about each:

| What they found | What we do about it |
|---|---|
| Testing only "does it catch hallucinations" hides a model that just says "wrong" to everything. | We always report scores for both classes separately (§5.4). We never publish one blended number. |
| Two human checkers agreed 91–93% of the time. | Our own target is 60% agreement as a **minimum**, not a goal. If we barely clear 60%, that means our labelling rules need work. |
| Code-mixed (Bengali + English) text behaves differently from pure Bengali. | That's exactly the question Phase 2 will study — which is why it's a separate phase, not folded into this one. |

**How we're different:** BenHalluEval tests how good *existing AI chatbots* are at spotting hallucinations, by asking them. We instead **train our own detector from scratch** and compare a full ladder of model types on it, across every school subject, without throwing away the hard cases (§5.5).

**A related Kaggle competition** ("অলীকবচন — Bengali LLM Hallucination Detection") may be where part of our source data originally came from. We haven't confirmed its licence yet — see open question Q1.

### 2.3 Why now

Bengali AI models (BanglaBERT, MuRIL, IndicBERT v2, and others) now exist. But nobody has trained and published a proper Bengali hallucination detector to test them on. This project does that.

---

## 3. Goals

### 3.1 The six main goals

| ID | Goal | Status |
|---|---|---|
| G1 | Build a checked Bengali dataset, at least 4,000 pairs | ✅ Done — 4,480 pairs, all annotated, 4,300 usable after cleanup |
| G2 | Include both "with passage" and "without passage" questions | ✅ Done — 60% with passage, 40% without |
| G3 | Test a full model ladder, from simple word-counting to pretrained transformers | ⬜ Not started |
| G4 | Score at least 0.80 on the hard has-context items, and 0.60 on no-context items | ⬜ Not started |
| G5 | Pass the shortcut check (no cheap trick works) | ✅ Passing |
| G6 | Keep everything repeatable and logged | ✅ In place |

**Why G4's target changed:** the original plan said "score 0.80 on all has-context items." But a dumb rule — "if the answer's words are in the passage, call it correct" — already scores 0.823 without understanding anything. So that target proved nothing. The real target is now the **hard** items, where that same dumb rule only scores 0.454 (§5.5).

### 3.2 What we are NOT doing in Phase 1 (saved for Phase 2)

- Anything in Banglish or romanised Bengali
- Studying how "code-mixing" (Bengali + English mixed together) affects results
- Marking *which specific words* in an answer are wrong
- Comparing AI-written vs human-written wrong answers
- Studying how well tokenizers handle Bengali
- Fine-tuning a full generative chatbot to be a detector
- Writing the final research paper

### 3.3 Things we must never do

- ❌ Make the score look better by weakening the dataset. A high score on a broken dataset is a failed project, not a good one.
- ❌ Report a has-context score without also showing the string-matching baseline next to it.
- ❌ Chase the newest, fanciest model just because it's trendy. The model list is fixed by the course.
- ❌ Let a model search or look anything up (§5.3a).
- ❌ Collect, scrape, or generate any more data. The dataset is finished (§5.1).
- ❌ Start Phase 2 work early.

---

## 4. Who cares about this project

| Role | Who | Cares most about |
|---|---|---|
| Owner | Tawhidul Hasan | Everything |
| Annotators | Owner + one more Bangla speaker | Labelling rules, the agreement score (§5.1a) |
| Course grader | Instructor | The full model ladder (§5.3), whether the lab syllabus was used (§5.3b), and repeatability |
| Future readers | Phase 2 team, reviewers | Whether the dataset is valid, licences are clear, and results are reported honestly |

---

## 5. Requirements

### 5.1 What the dataset must contain

**Every school subject is included — nothing is left out for being hard.** Law, science, BCS, and literature together make up 966 pairs (21.6% of the dataset), on equal footing with easier subjects. We measure difficulty; we don't hide it by deleting hard items.

**One exception — and it's about the *type* of question, not difficulty:** fill-in-the-blank questions are excluded. This project is Q&A only. A fill-in-the-blank question lets a model just copy words from the passage instead of actually judging an answer — a simple rule scores 0.929 on those alone. Removing them also removed the `geography` subject entirely, since every geography item was fill-in-the-blank. That's a side effect of the rule, not a judgement about geography.

**A past mistake, now fixed:** an earlier version of the dataset tried to filter out "hard to verify" items and accidentally removed about 4,150 pairs. **The project owner reversed that.** Instead of removing hard items, we now *measure* how hard each item is (the `difficulty` field) and report easy and hard separately (§5.5).

**The trade-off we accepted:** law, science, BCS, and literature questions often ask for a specific date, article number, or scientific name. A human checker can't always verify these from memory. §5.1b explains what checkers do instead. And because of this, every result **must** be broken down by subject, so a weak spot doesn't get hidden inside an average (§5.4).

**Only four other things get removed**, and none are about difficulty:
- Incomplete pairs (missing either a correct or a wrong answer)
- Text damaged by OCR scanning errors
- Questions that quote 6+ words straight out of their own passage (the question would answer itself)
- Duplicate questions (which would leak between train and test)

### 5.1b How checkers handle questions needing outside knowledge

| Type of question | Can the checker look it up online? |
|---|---|
| **No passage given** | **Yes.** Look it up, then label it, and note that you checked. |
| **Passage given** | **No.** Judge only against the passage. An outside source can't tell you "does *this* passage support *this* answer" — only the passage can. |

If a no-passage question still can't be settled even after checking, the checker writes `unsure` instead of guessing. `unsure` items don't count toward the agreement score, but we do report how many showed up per subject — a lot of `unsure` items in one subject is a finding about the dataset, not a sign the checker did badly.

**All requirements for the dataset:**

| ID | Requirement | Must-have? | Status |
|---|---|---|---|
| D1 | At least 4,000 pairs | Must | ✅ 4,480 |
| D2 | 60% with passage / 40% without | Must | ✅ 2,688 / 1,792 |
| D3 | Exactly 50/50 correct vs hallucinated | Must | ✅ Exact |
| D4 | Test set: 100% checked by two humans | Must | ✅ Done — agreement 0.865, disagreements resolved, merged in |
| D4a | Dev/train sheets may be pre-filled; test sheets may **not** (pre-filling test would leak the answer) | Must | ✅ Enforced by the sheet-building script |
| D5 | Human agreement score (Cohen's κ) of at least 0.60 | Must | ✅ 0.717 on the 100-item test run, ✅ 0.865 on the full 1,356-item test set |
| D6 | No subject excluded for being hard; Q&A only, no fill-in-the-blank | Must | ✅ 13 subjects included |
| D7 | No question ever appears in more than one of train/dev/test | Must | ✅ Verified |
| D8 | Every wrong answer in test, dev, and the 20% train sample has an error-type label (see §5.1c for why not *all* of train) | Must | ✅ Done — 1,867 of 1,977 items typed; the rest were excluded for good reason |
| D9 | Every pair marked easy or hard | Must | ✅ Done automatically, by rule |
| D10 | Licence recorded for every data source before use | Must | ⚠️ Two questions still open (see §12) |
| D11 | Extra fields for Phase 2 (`script_condition`, `cmi`) are filled in now | Should | ✅ Done |
| D12 | Mark exactly which words are wrong, where possible | Could | ⬜ Not done |
| D13 | No personal/private information in the released data | Must | ⬜ Scan not done yet |
| D14 | Pairs that human checkers found broken or wrongly labelled are excluded from both training and scoring | Must | ✅ 180 pairs excluded — see §5.1d |

### 5.1c Decision: error-type labels are only required for checked data

**Made 17 September 2026.**

**The old rule:** every wrong answer in the whole dataset (4,480 wrong answers) needs an error-type label.

**The new rule:** an error-type label is only required where a human actually checked the item — that's all of test (678), all of dev (673), and the 20% sample of train (626). That's **1,977** items total.

**Why this is fine, not a shortcut:**

| Reason | Explanation |
|---|---|
| The type label is just for *reporting* | It exists so we can say "the model misses numeric errors more than entity errors." No model is ever trained on this label — it would give away the answer. |
| We only ever report type breakdowns for dev and test | Nobody looks at a type breakdown for train. So a type label on an un-sampled train item would never be read by anyone. |
| The actual correct/wrong labels are untouched | This decision is only about the *type* label, not the *correct/wrong* label. We separately checked a sample of train's correct/wrong labels and found only 4.5% noise — well under the 5% threshold that would require a bigger check (see the guide, §5). |
| It saves real work for no real benefit | Labelling the other 2,503 train items would take about 2,500 more annotation actions and would not change a single number we report. |

**What this looks like in the data:** train items outside the sample keep `hallucination_type = unlabeled`, but their `type_source` field says `outside_train_sample` — so it's obvious this was on purpose, not something we forgot.

**When you write results:** always say your type breakdown covers dev/test, and that train's types are a 20% sample, not the full set.

### 5.1d Decision: some pairs are excluded from training and scoring

**Made 17 September 2026. This answers open question Q5.**

While checking the data, humans found some pairs where the stored label just can't be trusted. Those pairs are removed from both training and scoring.

| Why a pair is excluded | What it means | train | dev | test |
|---|---|---:|---:|---:|
| The "wrong" answer was actually correct | A reviewer looked again and confirmed it | 28 | 19 | 42 |
| The "correct" answer was called wrong by every checker who saw it | Every human who looked at it disagreed with the stored label | 28 | 34 | 17 |
| The item itself is broken | Unreadable or the question doesn't match the passage | 8 | 5 | 8 |
| **Total pairs excluded** (one pair can have more than one reason) | | **62** | **54** | **64** |
| **Pairs left to use** | | **3,067** | **619** | **614** |

**Why we did this:** if we score a model against a label that humans have confirmed is wrong, we're punishing the model for being right. And training on a wrong label just teaches the model the mistake. This rule was decided **before** any model was trained, using only human judgement — so nobody could tune it later to make a score look better.

**How it's applied, in practice:**

- We remove the **whole pair** (both the correct and wrong answer), never just one side. That keeps every split's balance at exactly 50/50.
- Nothing is deleted from the files. Both records get a flag, `excluded = true`, plus a note on *why*. So this decision can always be checked or reversed later.
- Every script that trains or scores a model must load data through `src/splits.py`, which automatically removes the excluded pairs. This is the *only* place that rule is applied, so it can never be applied inconsistently.
- The shortcut-check script (`src/audit.py`) runs on **both** the full data and the cleaned-up data. Both must pass: 0.534 and 0.514.

**What changes because of this:**

- The "does the answer's text appear in the passage" baseline score is now measured on the cleaned-up data: **0.823** overall, **0.987** on easy items, **0.454** on hard items (§5.5).
- Only checked items could be flagged this way. The 80% of train that was never checked still carries its own estimated ~4.5% noise (see guide §5).
- We never edit a label directly (§5.1a). Excluding pairs is itself something we report: how many, why, and from which split.

### 5.1a The correct/wrong label — this rule never changes

> **`1` = correct. `0` = hallucinated.**

This label answers one question: *"is this answer correct?"* Note that `1` means "good," which is the **opposite** of how most hallucination research papers do it (there, `1` usually means "bad, hallucinated").

- In a probability output, the second number (`probs[:, 1]`) is "how likely this is correct."
- The source data already used this convention, so **we never flip it, anywhere** — not in a script, not in a table.
- If a result table's numbers look swapped, the mistake is in how it's being read, not in the data.

### 5.2 Error-type categories

| If there's a passage (intrinsic) | If there's no passage (extrinsic) | If the answer is correct |
|---|---|---|
| `entity` — wrong person/place/thing | `fabricated` — made up | `none` |
| `numeric` — wrong number/date | `overclaim` — states something with no fixed answer as if certain | |
| `relational` — right things, wrong relationship | | |
| `contradiction` — says the opposite of the passage | | |

(`unlabeled` = not checked yet, or intentionally skipped — see §5.1c.)

### 5.3 The model ladder

The course requires a fixed set of models. **You may add extra models. You may not skip or replace a required one.**

The ladder follows the college's NLP lab course (Labs 1–5) step by step, then goes further with pretrained transformers. §5.3b explains exactly which lab topics map to which model, and which ones we chose not to use.

| ID | What it is | Comes from Lab |
|---|---|---|
| M1 | Simple word-counting models: Bag of Words and TF-IDF, feeding Naive Bayes, Logistic Regression, and a linear SVM | 2, 3 |
| M2 | Word-vector models: Skip-gram embeddings, averaged two ways, feeding Logistic Regression and XGBoost | 3 |
| M3 | Sequence models in PyTorch: plain RNN, BiRNN, stacked BiLSTM, and BiLSTM with attention | 4 |
| M4 | At least 5 pretrained transformer models, fine-tuned | (beyond the lab) |
| M5 | Must include BanglaBERT and MuRIL specifically | (beyond the lab) |
| M6 | Further pretrain at least one encoder (mBERT or XLM-R only — see the note below) | (beyond the lab) |
| M7 | Combine (ensemble) the top 3 models | (beyond the lab) |
| M8 | A zero-shot AI model as a reference point (at least 300 items) | (beyond the lab) |
| M9 | Try 3 different ways of formatting the input, on the best model | (beyond the lab) |
| M10 | Test whether cleanup steps like removing common words or stemming help or hurt | 1 |
| M11 | A character-level n-gram language model, used two ways: to score how "passage-like" an answer sounds, and as its own classifier | 2, 3 |
| M12 | Similarity features (edit distance, cosine similarity) between an answer and its own passage/question | 1, 3 |
| M13 | A Transformer built completely from scratch (no pretraining) — shows how much of M4's score comes from pretraining | 5 |
| M14 | Test whether shuffling word order breaks each model's predictions | 3 |

**A note on repeat runs:** models with no randomness (Naive Bayes, the n-gram language model, rule-based baselines) only need to run once. Every model that trains with randomness runs on all three seeds (42, 1337, 2024).

> **Reminder:** BanglaBERT is a different kind of model (an "ELECTRA discriminator"), not the usual kind you can further-pretrain with a standard masked-language-modelling method. Use mBERT or XLM-R for that step (M6) instead.

### 5.3b Which lab topics we use, and which we skip

The college NLP lab (folder `NLP Lab/`) teaches 33 topics across 5 labs. We checked every single one against five questions:

1. Does it help decide correct vs. hallucinated?
2. Does it actually work on Bengali?
3. Does it keep the evidence of an error (numbers, negation words, small spelling changes) instead of accidentally deleting it?
4. Does it follow our rules (no searching, no new data, don't touch the test set early)?
5. Would it teach us something worth reporting?

**26 topics passed and are used:**

| Lab | Topic | Used in |
|---|---|---|
| 1 | Cleaning text with regex | M10 — a Bengali-safe version; never deletes numbers or Bengali letters |
| 1 | Splitting text into words (tokenizing) | M10 — feeds M1–M3 and M11–M14 |
| 1 | Removing common words ("stop words") | M10 — only as a test, never the default |
| 1 | Stemming (cutting word endings) | M10 — only as a test, a Bengali version, never touching negation |
| 1 | Edit distance (how many letters differ between two words) | M12 features, and the fuzzy-match baseline (V6) |
| 2 | Bag of Words | M1 |
| 2 | TF-IDF | M1 |
| 2 | N-grams as model input | M1 |
| 2 | N-gram language model | M11 |
| 3 | Skip-gram word embeddings | M2 |
| 3 | Cosine similarity | M12 features, plus a sanity check on our word vectors |
| 3 | Naive Bayes with smoothing | M1; the same smoothing idea is reused in M11 |
| 3 | Logistic Regression | M1, M2, M12 |
| 3 | Averaging word vectors into a sentence vector | M2 |
| 3 | Weighting that average by TF-IDF | M2 |
| 3 | The "word order doesn't matter" limitation of simple models | M14 |
| 4 | Standard PyTorch training loop | M3, M13 |
| 4 | Loading pretrained word vectors into PyTorch | M3 |
| 4 | Plain RNN | M3 |
| 4 | Bidirectional RNN | M3 |
| 4 | Stacked BiLSTM | M3 |
| 4 | Attention mechanism | M3 |
| 4 | Handling padding correctly in loss calculations | M3, M13 |
| 5 | Building a Transformer from scratch | M13 |
| 5 | Positional encoding | M13 |
| 5 | Multi-head self-attention, padding mask, mean pooling | M13 |

**7 topics did not pass, and here's why:**

| Lab | Topic | Why we skipped it |
|---|---|---|
| 1 | Lemmatization (finding a word's dictionary form) | Needs a dictionary and grammar-tagging tool that doesn't reliably exist for Bengali. Stemming (which we do use) tests a similar idea. |
| 1 | Spelling correction | This would literally *fix* the errors we're trying to detect. We already exclude typo-only pairs elsewhere (§5.1d). |
| 2 | Text generation (guessing the next word) | Our project detects wrong answers. It doesn't generate new text — that would count as making new data, which is against the rules (§5.1). |
| 3 | Word analogies (king − man + woman = queen) | A fun demo, not something that detects hallucinations. We keep the useful part — checking our word vectors make sense — as a quick sanity check. |
| 4 | Generating text with an LSTM | Same reason as above: this is generation, not detection. |
| 4 | Tagging each word with a part of speech | Needs word-by-word labels our dataset doesn't have. Marking exactly which word is wrong is future work (Phase 2). |
| 4 | Translating one language to another (Seq2Seq) | There's no translation task here. Translating Banglish is Phase 2 work, and needs a proper transliteration tool first (§14.1). |

**Also decided (17 September 2026): we are not using outside pretrained Bengali word vectors (like fastText).** Instead, M2 and M3 train their own Skip-gram vectors using only the train split — which is exactly what the lab teaches.

### 5.3c Rules for adapting lab code to Bengali

The lab code was written for English. If you copy it exactly onto Bengali text, it silently breaks things:

| The lab's code | What it would do to Bengali text | The rule instead |
|---|---|---|
| `re.sub(r'[^A-Za-z\s]', '', text)` | Deletes **every single Bengali letter and every digit** | Only use Bengali-aware patterns. Always keep digits — 475 of our `numeric` error labels depend on them. |
| NLTK's English tools (`word_tokenize`, `stopwords`, `PorterStemmer`, `WordNetLemmatizer`) | Do nothing useful, or actively break Bengali text | Never use these on Bengali text. Use the Bengali versions described in the guide (§6). |
| Removing "stop words" | The published Bengali list we use contains negation words (না, নয়, নেই, নাই, ছাড়া) and number words (একটি, দুটি, হাজার, প্রথম). Removing negation turns "not divided" into "divided" — the opposite meaning — hiding our `contradiction` errors. Removing numbers hides `numeric` errors ("একটি" vs "দুটি"). | Negation and number words are listed in `configs/bn_protected_words.txt` and never removed. Stop-word removal is one experiment, never the default. |
| Stemming | A careless rule could strip the negation ending off a word like "করেননি" (did not do), turning it into "did do." | Never strip negation endings. Only test stemming as an experiment, never as the default. |

### 5.3a No searching or looking things up (RAG is forbidden)

**The model may never search for information at prediction time.** No vector database, no searching the training set for similar examples, no fetching an outside passage. The model sees only: the question, the passage already stored in that one record (if there is one), and the candidate answer. Nothing else.

**Why this rule exists:** if a model could search for facts, we'd be testing its search engine, not its ability to spot a wrong answer. And "no passage given" questions would become pointless, since the model could just search for the answer anyway.

**Everything else is allowed**, and trying extra techniques is encouraged — n-grams, word embeddings, RNNs, LSTMs (with or without attention), BERT-style models, combining models together, or anything else that isn't "searching." You can add to the required ladder in §5.3. You cannot remove from it.

**Important distinction: comparing something to itself is fine. Comparing it to *other* records is not.** In M12, we compare an answer to *its own* question and *its own* passage — that's allowed. In M11, we build a mini-language-model out of *that one record's own* passage — also allowed. But looking up *other* training examples to help answer a *different* question would be searching, and that's forbidden. (M11's second use — one trained model for "correct" answers, one for "wrong" answers — is a normal trained classifier, like Naive Bayes, not a lookup table.)

**Other places this rule could accidentally get broken:**

| Situation | What's allowed | What's forbidden |
|---|---|---|
| Prompting an AI model as a reference point (M8) | Zero-shot; or a **fixed** set of example answers shown for every single test item | Picking different example answers *for each* test item based on how similar they are — that's searching |
| Further pretraining (M6) | Feeding the model lots of unlabelled Bengali text *before* any testing happens | Fetching text *during* prediction |
| Combining models (M7) | Averaging several trained models' predictions | Weighting a model's vote based on how well it did on "similar" past examples |
| The annotation-helper script (`src/llm_annotate.py`) | It exists, but it's not part of any detector and has never been run | — |

**Checked on 17 September 2026:** there is no searching-related code or library anywhere in `src/`, `notebooks/`, `configs/`, or `requirements.txt`. The only script that connects to the internet is `src/llm_annotate.py`, and it has never been run.

### 5.4 Rules for reporting results

- The main score is **macro-F1** (treats both classes equally). Also report accuracy, precision/recall per class, and AUC.
- **Always** break results down by: with-passage vs. without-passage, easy vs. hard, subject, and error type. Error-type breakdowns only cover dev and test (§5.1c).
- Subject breakdowns are required because law, science, BCS, and literature need outside knowledge a closed-book model simply doesn't have. If the model does badly there, that's a real finding — report it, don't hide it.
- **Never** publish one single blended number. It hides exactly the differences that matter.
- Run every trainable model on all three seeds, and report the average and spread.
- Before claiming "model A beats model B," run a McNemar test and a bootstrap confidence interval.
- Models given *only the answer* (like M11's second classifier) should be compared to the "answer-only" sanity check (§5.5, V3). If they score suspiciously high, that means they learned the writing style of whatever wrote the wrong answers — not whether the answer is actually correct.
- The word-order test (M14) and the cleanup experiments (M10) are done on **dev only**. They inform decisions; they never touch the test set.

### 5.5 The validity checks (the safety gate)

| ID | What it checks | Status |
|---|---|---|
| V1 | A model given *only* surface features (like answer length) must score below 0.60 — on both the full data and the cleaned-up data | ✅ 0.534 / 0.514 |
| V2 | Re-run this check every time the dataset changes | ✅ Always done |
| V3 | A model given only the answer text should score clearly worse than a model given everything | ⬜ Needs a trained model |
| V4 | Every has-context score must be shown next to the string-matching baseline | ⬜ To enforce once models exist |
| V5 | Any score above 0.95 on has-context data triggers an investigation for a data leak | ⬜ |
| V6 | A "fuzzy" string-matching baseline (allows small spelling differences) must be shown next to the exact one | ⬜ To measure once models exist |

**V1 is a hard stop.** If you're ever tempted to lower the 0.60 threshold, change which features it looks at, or reshuffle the split just to make it pass — don't. That single move would be the most damaging thing anyone could do to this project.

**The exact string-matching baseline (V4):** "if the answer's exact text appears in the passage, call it correct."

| Group | Score on cleaned-up data (the number to beat) | Score on all data (before cleanup) |
|---|---:|---:|
| All has-context items | **0.823** (5,092 items) | 0.812 (5,376 items) |
| Easy items | 0.987 (3,398 items) | 0.980 (3,522 items) |
| **Hard items** | **0.454** (1,694 items) | 0.456 (1,854 items) |

Models are trained and scored only on the cleaned-up data, so the left column is the one to compare against. When you score dev or test specifically, recompute this baseline on that exact same set of items.

**Why we also need a "fuzzy" version (V6):** the exact-match rule fails the moment even one letter changes, which makes it look artificially bad on hard items (0.454). If a real model beats that number but *doesn't* beat a slightly smarter fuzzy-matching rule, it hasn't really learned to understand the passage — it's just doing fuzzy string-matching. Reporting both keeps us honest.

**Why hard items are hard, in one sentence:** correct answers in this data are usually copied word-for-word from the passage; wrong answers usually aren't. The `difficulty` field marks the exceptions — where that pattern doesn't hold — and that's exactly where a model has to actually read and understand.

### 5.6 Keeping everything repeatable

| ID | Requirement | Status |
|---|---|---|
| R1 | Every run (including failed ones) is logged | ✅ |
| R2 | Every seed is fixed and logged (42, 1337, 2024) | ✅ |
| R3 | Any AI prompts used are saved in version control | ⬜ |
| R4 | Labelling rules were written *before* labelling began | ✅ |
| R5 | Every library version is pinned in `requirements.txt` | ✅ |
| R6 | Re-running the split-building script gives byte-for-byte the same result | ✅ |

---

## 6. What counts as success

### 6.1 The main numbers we're chasing

| Metric | Target | Has to beat |
|---|---|---|
| Score on hard has-context items | **At least 0.80** | 0.454 (exact string-match rule, on cleaned data) |
| Score on all has-context items | Report it, no fixed target | 0.823 (exact string-match rule, on cleaned data) |
| Score on has-context items (all and hard) | Report it | The fuzzy string-match rule (V6) — will be measured once models exist |
| Score on no-context items | **At least 0.60** | 0.333 (always guessing the more common answer) |
| The safety-check score (V1) | **Below 0.60** | — |
| Human agreement score (κ) | **At least 0.60** | — |

### 6.2 Secondary goals

- Every required model on the ladder is tested and logged.
- Report the ranking honestly — including when two models tie.
- Full error-type breakdown, on dev and test (the annotation is done, see §5.1c).

### 6.3 How we compare to other published work

BenHalluEval reports a score between 7.72% and 55.42% across its models (lower is better there — it's an error rate, the opposite direction from macro-F1). **Our numbers are not directly comparable** — different task, different data, and we train models instead of just prompting them. Don't claim to have "beaten" them.

---

## 7. What's in scope, and what isn't

### 7.1 In scope

- Bengali script only
- School/exam subjects (BCS, SSC, HSC style), plus grammar and vocabulary
- Deciding correct-or-hallucinated for a single answer
- The fixed model ladder (§5.3)
- Both with-passage and without-passage questions

### 7.2 Out of scope (this phase)

- Anything Banglish or romanised (Phase 2)
- Marking exactly which words are wrong; measuring code-mixing; measuring tokenizer efficiency (Phase 2)
- Generating or fixing answers — this project only detects. That rules out the lab's text-generation, spelling-correction, and translation exercises too (§5.3b)
- Tagging individual words with labels — we don't have that kind of data
- Any language other than Bengali
- Running this as a live, production system

---

## 8. Milestones — the big checkpoints

| ID | Milestone | What it needs to pass | Status |
|---|---|---|---|
| M0 | Set up the project: repo, schema, seeds, raw data checked | Audit written | ✅ Done |
| M1 | Build the corpus, filter it, split it | Safety check score below 0.60 | ✅ Done — 0.534 |
| M2 | Prove the labelling process works | Human agreement ≥ 0.60 on 100 test items | ✅ Done — 0.717 |
| M3 | Fully annotate the corpus (test + dev + a train sample) | D4 and D8 both satisfied | ✅ Done — agreement 0.865; 253 disagreements resolved by hand; merged in; bad pairs excluded |
| M4 | Train and log the whole model ladder | Every required model logged, fuzzy baseline measured | ⬜ **This is next** |
| M5 | Extra experiments: more pretraining, combining models, AI reference point, word-order test | All checks from §5.5 enforced | ⬜ |
| M6 | Final scoring on the locked test set — done **exactly once** | Meets the targets in §6.1 | ⬜ |

---

## 9. What we will deliver

| ID | Deliverable |
|---|---|
| DL1 | The finished, annotated Bengali dataset |
| DL2 | Written labelling rules |
| DL3 | Human-agreement report — ✅ already written |
| DL4 | List of data sources and their licences |
| DL5 | Trained model files |
| DL6 | Result tables (see guide §12.4) |
| DL7 | Full log of every experiment run |
| DL8 | Safety-check report |
| DL9 | A codebase anyone can re-run |
| DL10 | The final written report for Phase 1 |

---

## 10. Risks and how we handle them

| ID | Risk | What we do about it |
|---|---|---|
| RK1 | A cheap trick inflates scores without real understanding | Run the safety check every time the data changes |
| RK2 | Human agreement drops below 0.60 | Fix the labelling *rules*, not blame the humans; re-test on the same 100 items |
| RK3 | Second checker drops out | Kept the test set small enough for one person to fully double-check alone |
| RK4 | The Kaggle dataset's licence turns out to forbid reuse | Keep those items clearly separable so they can be removed if needed |
| RK5 | Two models score too close to tell apart | Use 3 seeds plus a statistical test; report ties honestly |
| RK6 | No-passage questions score poorly | Expected — it's a genuinely harder task, not a bug |
| RK8 | Free GPU quota runs out | Use smaller models, shorter inputs, and efficient settings |
| RK9 | Team starts drifting into Phase 2 work early | Flag it immediately and stop |
| RK10 | The test set gets "used up" from being checked too many times | Use dev for everything; touch test exactly once, at M6 |
| RK11 | Hard-to-verify subjects hurt labelling quality | Accepted on purpose — checkers may verify or mark `unsure`; results are broken down by subject so any real effect is visible |

---

## 11. What this project depends on

**External:** Bengali Wikipedia (licence: CC BY-SA 4.0), BCS question banks (licence not yet confirmed), the models BanglaBERT/MuRIL/XLM-R/mBERT, a free GPU from Kaggle or Colab.

**Internal:** the second human checker, and the project owner's own review of the dataset.

---

## 12. Questions still open

| ID | Question | What it's blocking |
|---|---|---|
| Q1 | What licence covers the Kaggle competition data, if we used any of it? | Releasing the dataset publicly |
| Q2 | Did the BCS questions come from an official government source, or a commercial book? | Releasing those specific items |
| Q3 | What licence should our finished, released dataset carry? | Releasing the dataset publicly |
| Q4 | Which exact AI model and prompts were used to originally write the source Q&A pairs? | Full reproducibility |
| Q5 | ~~How should we treat the 180 pairs that human review flagged as bad?~~ **Answered on 17 September 2026 — see §5.1d.** | — |

---

## 13. When is Phase 1 actually finished?

All of these must be true:

1. ✅ `train.jsonl`, `dev.jsonl`, and `test.jsonl` are built, locked, and fully annotated.
2. ✅ The safety check passes, and the report proving it is saved.
3. ⬜ Every required model (M1–M14) has been run — with 3 seeds wherever randomness is involved — and logged.
4. ⬜ Hard has-context score is at least 0.80, and no-context score is at least 0.60 — or, if not, a clear written explanation of why not.
5. ⬜ Every has-context score in every table is shown next to both string-matching baselines.
6. ⬜ The test set has been scored exactly once.
7. ⬜ Every licence question is resolved.
8. ✅ Cloning the repo fresh and re-running the scripts reproduces the exact same dataset.

---

## 14. What happens after Phase 1 (Phase 2 preview)

Phase 2 studies Banglish: build a code-mixed dataset, run the same pipeline, and measure how mixing Bengali and English hurts detection.

### 14.1 What went wrong the first time we tried Banglish — read before starting Phase 2

The project originally tried to start with Banglish. We used a **hand-written letter-conversion tool** to turn Bengali into romanised text. The output was so broken that **a native Bangla speaker couldn't even read it**, and the 500-item test batch built on it had to be thrown away entirely.

**The lesson:** don't build your own letter-by-letter conversion tool. Phase 2 must use a real, tested transliteration tool, and every batch must be checked by a native speaker *before* anyone starts labelling it.

### 14.2 What Phase 1 hands over to Phase 2

- A working, tested pipeline that just needs different input data
- A dataset format that already has slots ready for Phase 2's extra fields
- Labelling rules already proven to reach a 0.60+ agreement score
- Bengali baseline scores to compare Phase 2's code-mixed scores against

---

## Glossary

| Term | Meaning |
|---|---|
| **has-context** | The question comes with a passage; the answer must be supported by it |
| **no-context** | No passage is given; the model must answer from what it already knows |
| **hard** | An item where the simple "check if the words match" trick doesn't work |
| **pair** | One question, with one correct answer and one wrong answer |
| **intrinsic error** | The answer contradicts the given passage |
| **extrinsic error** | There's no passage to check against, so the answer just has to be true or false on its own |
| **shortcut** | A cheap trick that predicts the right answer without actually solving the task |
| **Banglish** | Bengali written using English letters — Phase 2 only |
