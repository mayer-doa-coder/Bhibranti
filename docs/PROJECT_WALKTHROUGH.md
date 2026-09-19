# BangHallu — Project Walkthrough (Phase 1)

**What this file is:** a plain-language explanation of the whole project, step by step — what we
did, why we did it, how it works, and what came out. It is written so that someone who has never
seen the code (a teacher, a classmate, or ourselves a month from now) can follow it and explain it.

**Who:** Tawhidul Hasan (2107004) · Md. Saif Ahmed Shejan (2107009)
**Supervisors:** Dr. K. M. Azharul Hasan · Md. Nazirulhasan Shawon
**Covers:** the beginning of the project → the end of the data phase (annotation complete and
merged), plus the plan for the model ladder built on the NLP lab. **Last updated:** 17 September 2026.

> **Keep this file alive.** Each time a step finishes, add a new section in the same format
> (What / Why / How / Result / Checked?) and update the status table in §2.

Every number in this file was re-measured from the actual data on the date above, not copied from
older notes. Where an older document had a wrong number, §15 says so.

---

## Contents

1. [The project in one minute](#1-the-project-in-one-minute)
2. [Where we are — the status table](#2-where-we-are--the-status-table)
3. [Words you need (glossary)](#3-words-you-need-glossary)
4. [Step 0 — Choosing the scope: Bengali first](#4-step-0--choosing-the-scope-bengali-first)
5. [Step 1 — Foundations: one record format, one label rule, fixed seeds](#5-step-1--foundations)
6. [Step 2 — Getting the raw data and auditing it](#6-step-2--getting-the-raw-data-and-auditing-it)
7. [Step 3 — Cleaning](#7-step-3--cleaning)
8. [Step 4 — Building the corpus](#8-step-4--building-the-corpus)
9. [Step 5 — Measuring difficulty (easy vs hard)](#9-step-5--measuring-difficulty-easy-vs-hard)
10. [Step 6 — Splitting into train, dev and test](#10-step-6--splitting-into-train-dev-and-test)
11. [Step 7 — The shortcut audit](#11-step-7--the-shortcut-audit)
12. [Step 8 — Annotation guidelines and the 100-item agreement test](#12-step-8--annotation-guidelines-and-the-100-item-agreement-test)
13. [Step 9 — Full human annotation](#13-step-9--full-human-annotation)
14. [Step 10 — Merging the annotation into the corpus](#14-step-10--merging-the-annotation-into-the-corpus)
15. [Health check — was each step done correctly?](#15-health-check--was-each-step-done-correctly)
16. [All the parameters in one place](#16-all-the-parameters-in-one-place)
17. [Questions a teacher is likely to ask](#17-questions-a-teacher-is-likely-to-ask)
18. [What comes next](#18-what-comes-next)

---

## 1. The project in one minute

**The problem.** AI chatbots (large language models, LLMs) answer Bengali questions fluently and
confidently — and are sometimes simply wrong. A confidently wrong answer is called a
**hallucination**. For a student preparing for SSC, HSC or BCS exams this is dangerous: the answer
*looks* right, so the student memorises a false fact.

**What we build.** A program — a **detector** — that reads:

- a **question**,
- sometimes a **passage** of text the answer should come from, and
- a **candidate answer**,

and decides: **is this answer correct (`1`) or hallucinated (`0`)?**

**A real example from our data:**

> **Passage:** এ বিদ্রোহের সঙ্গে জড়িত থাকার অভিযোগে আলাওল **পঞ্চাশ দিন** কারাভোগ করেন …
> **Question:** শাহ সুজার বিদ্রোহে জড়িত থাকার অভিযোগে আলাওল কতদিন কারাভোগ করেন?
>
> | Candidate answer | Detector should say |
> |---|---|
> | পঞ্চাশ দিন | `1` correct — the passage says so |
> | একশ দিন | `0` hallucinated — the passage says fifty, not a hundred |

**What Phase 1 delivers.**

1. A clean, **human-checked Bengali dataset** (we call it the *corpus*) of 4,480 question–answer
   pairs, 4,300 of them usable after human review.
2. A **ladder of models**, from very simple (word counting) to modern (BERT-style transformers),
   climbing through the NLP lab syllabus (Labs 1–5), all tested on that dataset so we can compare
   them fairly.
3. A **reproducible pipeline**: anyone can rerun our scripts and get byte-for-byte the same data.

**Done so far:** (1) is complete — built, annotated, adjudicated and merged. (3) is in place for the
data. (2), the models, is next.

**The problem never changed.** From the first day it has been: *given a question, maybe a passage,
and an answer — is the answer correct or hallucinated, without looking anything up?* Adding the lab
topics changed **how** we study it (more model types, and tests of *why* each works), not **what**
we solve.

**What Phase 1 will be able to tell you, with numbers:**

| Question | How we find out |
|---|---|
| How accurately can a model spot wrong Bengali answers, with and without a passage? | every model, scored separately for the two cases |
| Does it really read the passage, or just check if the words appear there? | the hard subset, compared with an exact and a fuzzy string matcher |
| How far does each lab technique get — word counts, n-gram language models, word vectors, RNNs, a Transformer? | the model ladder, in lab order |
| How much of the best score comes from **pretraining**? | a Transformer trained from scratch vs BanglaBERT |
| Do English-style cleaning steps (stop words, stemming) help or hurt in Bengali? | the preprocessing variants |
| Which models actually use **word order**? | the word-order shuffle test |
| Which error types and subjects are hardest? | breakdowns by type and subject |
| Can we trust the labels? | κ = 0.865, 4.5% measured noise, 180 unusable pairs removed |

---

## 2. Where we are — the status table

| # | Step | Status | Key result |
|---|---|---|---|
| 0 | Choose scope: Bengali first, Banglish later | ✅ done | Phase 1 = Bangla script only |
| 1 | Foundations: schema, label rule, seeds, experiment log | ✅ done | `1 = correct`, seeds 42 / 1337 / 2024 |
| 2 | Get raw data and audit it | ✅ done | 62,084 raw records, 14 subject files |
| 3 | Clean | ✅ done | 56,480 records after removing duplicates |
| 4 | Build the corpus | ✅ done | **4,480 pairs / 8,960 records** |
| 5 | Measure difficulty | ✅ done | every pair marked easy or hard |
| 6 | Split into train / dev / test | ✅ done | 3,129 / 673 / 678 pairs, no overlap |
| 7 | Shortcut audit (the gate) (a cheap trick (like just checking answer length) could guess the correct answer without real understanding) | ✅ passing | metadata probe **0.534** (must be < 0.60) |
| 8 | Guidelines + 100-item agreement test | ✅ passed | Cohen's κ **0.717** (must be ≥ 0.60) |
| 9 | Full human annotation | ✅ done | 5,310 rows; test κ **0.865** |
| 10 | Merge annotation into the corpus | ✅ done | 253 rows adjudicated (a disagreement got a final decision from a third look); D8 met; strict audit passes |
| 10b | Leave out human-flagged pairs (PRD Q5) | ✅ done | 180 pairs excluded; usable: 3,067 / 619 / 614 pairs |
| 11 | **Model ladder (N-gram → … → BERT)** | 🔶 **in progress** | step 1 done: Bengali text tools (§18.3) |
| 12 | Final test evaluation (once only) | ⬜ not started | — |

In the project requirements document ([PRD.md](PRD.md)) these are milestones **M0** (steps 0–3),
**M1** (steps 4–7), **M2** (step 8), **M3** (steps 9–10), **M4–M5** (step 11), **M6** (step 12).

---

## 3. Words you need (glossary)

| Word | Simple meaning |
|---|---|
| **Hallucination** | An answer that sounds confident but is wrong or not supported. |
| **Record** | One row of data: one question + (maybe) a passage + **one** answer + its label. |
| **Pair** | Two records sharing the same question: one with a correct answer, one with a wrong answer. Our corpus is built entirely out of pairs. |
| **Label** | The truth for a record. **`1` = correct, `0` = hallucinated.** |
| **has-context** | A passage is given. Question to ask: *does the passage support this answer?* |
| **no-context** | No passage (closed-book). Question to ask: *is this answer true?* |
| **Corpus** | Our final dataset: `data/corpus/bn_v1/corpus.jsonl`. |
| **Train / dev / test** | Three non-overlapping parts of the corpus: to learn from, to tune on, and to grade once at the end. |
| **Shortcut** | A cheap trick that predicts the label *without* doing the real task (e.g. "long answers are usually wrong"). A model that learns a shortcut looks good and is useless. |
| **Probe** | A deliberately dumb model we train to *find* shortcuts. If a dumb model scores well, the data is leaking the answer. |
| **Macro-F1** | Our main score, from 0 to 1. It averages how well the model does on the "correct" class and on the "hallucinated" class, so it cannot be gamed by always guessing one side. Random guessing ≈ 0.50; always guessing one class = 0.33. |
| **Annotation** | Humans reading records and writing down their own judgement. |
| **Cohen's κ (kappa)** | How much two annotators agree **beyond what luck alone would give**. 0 = no better than chance, 1 = perfect. |
| **Seed** | A fixed starting number for anything random, so the "random" result is the same every time. |
| **Hallucination type** | *What kind* of wrong a wrong answer is (wrong name, wrong number, …). See §13. |
| **RAG** | Retrieval-augmented generation: letting a model look facts up. **Forbidden** in this project (§5). |

---

## 4. Step 0 — Choosing the scope: Bengali first

**What.** Phase 1 works only on **Bengali script** (বাংলা হরফ). **Banglish** (Bangla written in
English letters, e.g. *"Alaol koto din karabhog koren?"*) is postponed to Phase 2.

**Why.** The project originally started with Banglish. To create Banglish data, we converted
Bengali text using **hand-written letter-by-letter rules**. The output was unreadable even to a
native Bangla speaker, so the 500-pair pilot built on it had to be thrown away.

Two lessons came out of that:

1. Banglish detection is a *harder problem stacked on top of* Bengali detection. If we cannot
   detect hallucinations in clean Bengali, we certainly cannot in Banglish. So solve Bengali first.
2. Phase 2 must use a proper transliteration tool, and a native speaker must read the output
   before anything is built on it.

**Other scope decisions made at the start** (all in [PRD.md](PRD.md) §3):

| Decision | Reason |
|---|---|
| **QA only** — no fill-in-the-blank items | A fill-in-the-blank item trains "copy the missing words", not "check if an answer is right" (§8). |
| **No subject is removed for being hard** | Law, science, BCS and literature need outside knowledge. Removing them would make the task look easier than it really is. We *measure* difficulty instead (§9). |
| **No RAG** (no looking things up) | If a model could search for facts, we would be measuring its search engine, not its ability to spot a wrong answer — and the no-context condition would become meaningless. |
| **No new data collection** | The corpus is final. Effort goes into quality, not quantity. |
| **Fixed model ladder** | Required by the course: N-gram → Skip-gram/word2vec → BiRNN/BiLSTM → BERT-family. We may add models but not replace these. The ladder follows the NLP lab (Labs 1–5) step by step — see §18. |

---

## 5. Step 1 — Foundations

Before touching data we fixed four things that must never change later.

### 5.1 One record format (the *schema*)

File: [`configs/schema.json`](../configs/schema.json). Every record has the same fields. The
important ones:

| Field | Example | Meaning |
|---|---|---|
| `id` | `bnp_000651_1` | unique record id |
| `pair_id` | `bnp_000651` | links the correct and wrong answer of one question |
| `subject` | `history` | topic |
| `condition` | `has_context` | passage given or not |
| `context` | "এ বিদ্রোহের সঙ্গে…" | the passage (empty for no-context) |
| `question` | "…আলাওল কতদিন কারাভোগ করেন?" | the question |
| `candidate_answer` | "একশ দিন" | the answer being judged |
| `label` | `0` | **the truth** (1 correct, 0 hallucinated) |
| `difficulty` | `easy` / `hard` | see §9 |
| `hallucination_type` | `numeric` | kind of error (see §13) |
| `annotator_1`, `annotator_2` | `0`, `0` | what each human said |

**Why freeze it?** Every script reads these fields. If one script renamed a field, everything
downstream would silently break. Fields may be *added*, never renamed or removed. Some fields
(`cmi`, `script_condition`, `error_span`) are there only for Phase 2 — we fill them now so we do
not have to rebuild later.

### 5.2 One label rule, everywhere

> **`1` = correct. `0` = hallucinated.**

Many research papers use the opposite (1 = hallucinated). We kept `1 = correct` because our source
files already used it, so **no label is ever flipped anywhere** — flipping in one place and
forgetting in another is one of the most common and hardest-to-notice bugs in this kind of work.

### 5.3 Fixed random seeds: 42, 1337, 2024

Shuffling, sampling and model training all use randomness. Fixing the seed means rerunning a script
gives *exactly* the same output. Every model will be trained three times (once per seed) and we
report the average ± spread, so one lucky run cannot mislead us.

### 5.4 One experiment log

[`results/experiment_log.csv`](../results/experiment_log.csv) records **every** run — including
failures — with its date, settings, seed and score. It currently has 12 rows (audits, baselines and
the two κ measurements).

---

## 6. Step 2 — Getting the raw data and auditing it

**What.** The starting material is `data/raw/bn_qa_pool/`: **14 files, one per subject, 62,084
records**, each shaped like `{context, prompt_bn, response_bn, label}`.

| Subject file | Records | | Subject file | Records |
|---|---:|---|---|---:|
| mathematics | 29,332 | | science | 4,000 |
| grammar | 5,000 | | bcs | 2,324 |
| history | 5,000 | | antonym / idiom_meaning / vocabulary | 2,000 each |
| reading_comprehension | 5,000 | | geography | 1,741 |
| law | 1,383 | | synonym | 1,000 |
| literature | 914 | | others | 390 |

**Where it comes from.** The text comes from **Bengali Wikipedia** (licence CC BY-SA 4.0) and **BCS
question banks**. The question–answer pairs were built from those texts with LLM assistance.
Details and two unresolved licence questions are in [`data/SOURCES.md`](../data/SOURCES.md).

**Why audit before using it?** You cannot trust a dataset you have not measured. The full audit is
[`data/DATASET_AUDIT.md`](../data/DATASET_AUDIT.md). What it established:

| Finding | Evidence | Consequence |
|---|---|---|
| **Label meaning confirmed: `1` = correct** | In has-context records, 78.6% of label-1 answers appear word-for-word in the passage vs only 6.9% of label-0 answers; spot checks agree (২৬ মার্চ = national day ✓, ১৬ ডিসেম্বর ✗) | no flipping needed (§5.2) |
| It really is Bengali script | Latin letters are only 0.96% of all letters | fits Phase 1 |
| Heavily unbalanced subjects | mathematics is 47% of all records | a subject cap is needed (§8) |
| Mostly closed-book | 80% of records have no passage | has-context data is the scarce resource |
| **Duplicates** | 5,536 exact duplicates (8.9%) | must be removed (§7) |
| **A severe shortcut exists** | correct answers are usually copied from the passage, wrong ones are not | must be measured and reported (§9, §11) |

---

## 7. Step 3 — Cleaning

**Script:** [`src/build_bn_pool.py`](../src/build_bn_pool.py) → **Output:**
`data/interim/bn_pool.jsonl`

**What it does.**

1. Converts each raw record into our schema (§5.1).
2. Removes **5,536 exact duplicates** — the same record appearing twice.
3. Removes **68 contradictory records** — identical text carrying *both* label 1 and label 0 (the
   data disagrees with itself, so neither copy can be trusted).

**Result:** 62,084 − 5,536 − 68 = **56,480 clean records**.

**Why duplicates matter so much.** If the same question sits in both the training data and the test
data, a model can score well by *remembering* it instead of *reasoning* about it. That is called
**leakage**, and it produces scores that look great and mean nothing.

---

## 8. Step 4 — Building the corpus

**Script:** [`src/build_corpus.py`](../src/build_corpus.py) → **Output:**
`data/corpus/bn_v1/corpus.jsonl` and `data/splits/`

### 8.1 Why pairs?

The corpus is made of **pairs**: for each question, exactly one correct answer and one wrong answer.

This guarantees a perfect **50/50 balance** of labels, and — more importantly — it forces the model
to judge the *answer*. With pairs, the question alone tells you nothing about the label, because
every question appears once with each label.

### 8.2 The funnel — how 56,480 records became 4,480 pairs

| Stage | Pairs | Removed | Why |
|---|---:|---:|---|
| Complete pairs in the clean pool | 16,915 | — | records without a partner answer cannot form a pair |
| − fill-in-the-blank questions | | 816 | **QA only.** A string matcher already scores 0.929 on them; they teach copying, not checking |
| − OCR-damaged text | | 8 | broken vowel signs (e.g. "বিষয়ের ি") — unreadable, so un-labelable |
| − question copied from its own passage (6+ words in a row) | | 1 | such a question gives its own answer away |
| After validity filters | 16,090 | | |
| − repeated question text | | 159 | one copy could land in train and the other in test = leakage |
| **Available** | **15,931** | | 2,688 has-context, 13,243 no-context |
| **Selected into the corpus** | **4,480** | | all 2,688 has-context + 1,792 no-context |

**Removing fill-in-the-blank also removed `geography` completely** — every geography item was
fill-in-the-blank. That is a consequence of the task rule, not a judgement that geography is too
hard.

**What is deliberately *not* removed:**

- **Hard subjects.** An earlier version filtered out ~4,150 "hard to verify" pairs (law, science,
  BCS, literature). We **reversed** that decision: removing hard items would make the benchmark
  easier than reality. Those four subjects now make up **966 pairs (21.6%)**.
- **Pairs where the wrong answer is *also* copied from the passage.** These are exactly the pairs
  a string-matching trick cannot solve (§9), so they are the most valuable ones.

### 8.3 Composition rules

| Rule | Value | Why |
|---|---|---|
| has-context : no-context | **60 : 40** (2,688 : 1,792 pairs) | The grounded task is the main one, but closed-book matters too. Has-context data is scarce, so *all* of it is used. |
| Label balance | **50 : 50** (exact) | guaranteed by pairs |
| Max share of one subject | **50%** in has-context, **30%** in no-context | Without a cap, mathematics (59% of the no-context pool) would turn the benchmark into an arithmetic test. Has-context gets a looser cap because its two biggest subjects (reading comprehension, history) are the same kind of task and the data is scarce. |

### 8.4 What the corpus looks like

| Condition | Subjects (pairs) |
|---|---|
| **has-context** (2,688) | reading_comprehension 1,261 · history 880 · law 228 · grammar 169 · others 150 |
| **no-context** (1,792) | bcs 212 · antonym 212 · mathematics 211 · grammar 211 · idiom_meaning 211 · science 211 · literature 211 · law 104 · vocabulary 103 · synonym 73 · history 33 |

13 subjects in total. Passages have a median length of 266 characters (longest: 3,132). Questions
have a median of 54 characters; answers a median of 16.

**Reproducible:** running the script again produces byte-identical files (re-verified by SHA-256
hash on 17 September 2026).

---

## 9. Step 5 — Measuring difficulty (easy vs hard)

### 9.1 The problem: a string-matching shortcut

In the source data, correct answers are usually **copied straight out of the passage**, and wrong
answers usually are not. So this dumb rule —

> *"If the answer's text appears in the passage, say correct; otherwise say wrong."*

— already scores **macro-F1 = 0.823** on the usable has-context records (0.812 on all of them),
**without understanding anything**.

That means a real model scoring 0.82 has learned almost nothing beyond string matching. We need a
way to see past this.

### 9.2 The solution: mark every pair easy or hard

**has-context pairs:**

- **Easy** = the string rule *separates* the two answers (one appears in the passage, the other
  does not). The shortcut works.
- **Hard** = the string rule *cannot* separate them (both appear in the passage, or neither does).
  The model must actually read.

> **Hard example (from train):**
> **Passage:** কান্তনগর মন্দিরের নির্মাণ **১৭০৪** সালে শুরু হয় এবং **১৭৫২** সালে সম্পন্ন হয়।
> **Question:** কান্তনগর মন্দিরের নির্মাণ কাজ কত সালে শুরু হয়?
> Correct: ১৭০৪ সাল · Wrong: ১৭৫২ সাল
> **Both years are in the passage**, so "does it appear?" is useless. Only understanding *শুরু*
> (start) vs *সম্পন্ন* (finish) gives the answer.

**no-context pairs** (no passage, so the string rule does not apply):

- **Hard** = the wrong answer is a **near-miss** — at least 60% similar in characters to the
  correct one.
- **Easy** = the wrong answer is obviously different.

> **Hard example:** ‘প্রত্যাবর্তন’ শব্দের সন্ধি বিচ্ছেদ কী? Correct: প্রতি + আবর্তন · Wrong: প্রতি + আবর্তণ
> **Easy example:** জসীম উদ্‌দীনকে কোন অভিধায় ভূষিত করা হয়? Correct: পল্লিকবি · Wrong: ভোরের পাখি

### 9.3 Result

| Condition | Hard pairs | Easy pairs |
|---|---:|---:|
| has-context | 927 (34%) | 1,761 (66%) |
| no-context | 407 (23%) | 1,385 (77%) |

And the string-matching rule on has-context **records**:

| Slice | Usable records (§14.6) — **reference** | All records | What it tells us |
|---|---:|---:|---|
| All has-context | **0.823** (5,092) | 0.812 (5,376) | the number to beat, but it is inflated |
| Easy | 0.987 (3,398) | 0.980 (3,522) | the shortcut solves these almost perfectly |
| **Hard** | **0.454** (1,694) | 0.456 (1,854) | **worse than guessing** — the shortcut is useless here |

The "usable" column leaves out 180 pairs that human review found broken or wrongly labelled
(§14.6). Models train and are scored on that data, so those are the numbers to compare with.

**Why this matters for the whole project:** our target is **macro-F1 ≥ 0.80 on the hard subset**,
not on all items. Every has-context score we ever report will have the string-matcher score for
the same records (0.823 and 0.454 on the usable data) written next to it, so no one — including
us — can be fooled by the shortcut.

**Why difficulty is *measured* and not *filtered*:** we keep both easy and hard items and report
them separately. The *gap* between easy and hard is itself one of the findings.

---

## 10. Step 6 — Splitting into train, dev and test

### 10.1 Why three parts? (the exam analogy)

| Part | Analogy | Used for | How often |
|---|---|---|---|
| **Train** (70%) | the textbook | the model learns from it | constantly |
| **Dev** (15%) | practice tests | choosing settings, comparing models, catching mistakes | as often as needed |
| **Test** (15%) | the final exam | the one honest score reported at the end | **exactly once** |

If you practise on the final exam paper, your final score no longer measures anything. Every
decision made after looking at the test set "uses up" a little of its honesty. That is why
`test.jsonl` is locked until the very last step.

### 10.2 The result

| Split | Pairs | Records | has-context share |
|---|---:|---:|---:|
| train | 3,129 | 6,258 | 60.1% |
| dev | 673 | 1,346 | 59.7% |
| test | 678 | 1,356 | 59.9% |

### 10.3 How the split was done — and why it is done that way

1. **Split by pair, never by record.** If we split records, the correct answer to a question could
   land in train and its wrong answer in test. The model could then "remember" the question from
   training and score without judging anything. Both answers of a question always go to the same
   split.
2. **Split inside each (condition, subject) group.** Each group — e.g. "has-context history" — is
   shuffled and cut 70/15/15 on its own. That is why all three splits have the same 60/40
   condition mix and the same subject mix, so dev and test are fair miniatures of train.
3. **Fixed seed 42**, so the split is identical every time.

**Verified:** 0 pairs and 0 identical question texts are shared between any two splits.

---

## 11. Step 7 — The shortcut audit

**Script:** [`src/audit.py`](../src/audit.py) · **This is a blocking gate:** if it fails, nothing
built on the data can be trusted.

**The idea.** Before trusting any clever model, we train deliberately *stupid* models that are
forbidden from seeing the real content. If one of them scores well, the data is giving the answer
away through some side channel, and a real model would learn that side channel too.

| Probe | What it is allowed to see | Score | Verdict |
|---|---|---:|---|
| **Metadata-only** (the gate) | only surface features of the answer: word count, character count, average word length, share of Latin letters, counts of `, . ? ! "`, share of digits, share of capitals | **0.534** all · **0.514** usable | ✅ **PASS** on both (must be < 0.60; < 0.55 counts as "clean") |
| Answer-only character n-grams | the answer text, but no question and no passage | 0.542 | mild signal only — good |
| String matcher | "is the answer in the passage?" | 0.812 all · 0.823 usable | the shortcut from §9 — reported, not hidden |
| Majority class | always predicts the same label | 0.333 | sanity floor |

**How to read 0.534:** a model that only sees the *shape* of the answer — how long it is, whether it
has digits — does only slightly better than a coin flip (0.50). So the label is **not** leaking
through answer length or formatting.

**Why 0.60 and why it can never be relaxed.** If we loosened the threshold, removed features, or
changed the split just to make the gate pass, we would be hiding the exact problem the gate exists
to catch. If it ever fails, the data gets fixed and rebuilt — never the gate.

The audit is rerun after every change to the corpus. It was last rerun on 17 September 2026:
still **0.534, PASS**.

---

## 12. Step 8 — Annotation guidelines and the 100-item agreement test

### 12.1 Why humans at all?

The labels came with the source data, which was built with LLM help. A hallucination detector
trained on labels nobody checked might just learn the mistakes of whatever produced them. Humans
verify the labels and add the error type.

But humans disagree too — so before annotating thousands of rows, we must prove that **two people
following the same rules reach the same answer**. Otherwise the "human labels" are just one
person's opinion.

### 12.2 The guidelines

File: [`docs/ANNOTATION_GUIDELINES.md`](ANNOTATION_GUIDELINES.md), written **before** annotation
began.

**The one question:** *is this answer correct or wrong?* — asked differently for each condition:

| Condition | Ask | May you look it up? |
|---|---|---|
| has-context | "Does **this passage** support the answer?" (not "is it true in real life?") | **No.** An outside source cannot tell you what the passage says. |
| no-context | "Is this answer true?" | **Yes**, and note that you checked. |

If an item genuinely cannot be labelled (broken text, the passage does not answer the question),
the annotator writes **`unsure`** instead of guessing. `unsure` items are left out of κ and reported
by subject.

The guidelines contain **15 rules** for confusing cases, for example:

- **Rule 1** — different words, same meaning = still correct (এবং vs ও).
- **Rule 2** — right topic, wrong fact = wrong (বিজারক vs জারক).
- **Rule 3** — partly correct = wrong.
- **Rule 11–15** — added *after* the agreement test, from the cases where we actually disagreed
  (e.g. the passage does not answer the question; ignore `[1]` citation marks; শাব্দিক অর্থ means
  the literal meaning).

### 12.3 The agreement test (milestone M2)

**Script:** [`src/build_agreement_test.py`](../src/build_agreement_test.py) → 100 items.

**How it was designed.** The 100 items copy the corpus's mix: 60 has-context / 40 no-context, 50
correct / 50 wrong, and about the corpus's share of hard items. Both annotators labelled
**independently and blind** — no answer key, no comparing.

**Results** ([IAA report](../data/annotated/agreement_test_v1/IAA_REPORT.md)):

| Measure | Value |
|---|---|
| Items scored | 92 of 100 (8 marked `unsure`) |
| Raw agreement | 79 / 92 = 85.9% |
| **Cohen's κ** | **0.717** ✅ (needed ≥ 0.60) |

### 12.4 How Cohen's κ works (with our numbers)

Raw agreement flatters. On a yes/no task where both people say "yes" about half the time, they will
agree about **50% of the time by pure luck**.

κ removes that luck:

```
κ = (observed agreement − chance agreement) / (1 − chance agreement)
```

With our pilot: observed = 0.859, chance ≈ 0.502

```
κ = (0.859 − 0.502) / (1 − 0.502) = 0.357 / 0.498 ≈ 0.717
```

Read it as: *"of all the agreement that was possible beyond luck, we achieved 72% of it."*

| κ | Common interpretation (Landis & Koch) |
|---|---|
| < 0.40 | poor to fair — the guidelines are broken |
| 0.41 – 0.60 | moderate |
| **0.61 – 0.80** | **substantial** ← pilot, 0.717 |
| **0.81 – 1.00** | **almost perfect** ← full test split, 0.865 (§13) |

### 12.5 What the 8 `unsure` items taught us

They were **defects in the data**, not annotator weakness: three `law` items where the passage did
not actually answer the question, two where the answer type did not match the question (asked
*when*, answered with a biography), and three with broken text. Measured across the corpus, this
kind of passage mismatch affects roughly 1% of has-context items.

---

## 13. Step 9 — Full human annotation

### 13.1 What gets annotated, and how much

| Part | What | Who | Why this much |
|---|---|---|---|
| **test** (1,356 records) | **100%**, twice | **both** annotators, independently and blind | the final score is computed on it, so it must be as trustworthy as possible |
| **dev** (1,346) | 100% | one annotator | used for all tuning decisions |
| **train** (1,252 of 6,258) | a **20% random sample** (626 pairs) | one annotator | models tolerate a little noise in training data; the sample tells us whether train is sound |

### 13.2 The annotation sheets and the traps they avoid

**Script:** [`src/build_annotation_sheets.py`](../src/build_annotation_sheets.py) → spreadsheets in
`data/annotated/round1/`, 250 rows per file, 6 files per set.

Each row asks for: `your_label` (correct / wrong / unsure / unreadable), `your_type`,
`your_difficulty`, `your_notes`.

Every safeguard exists because the problem actually happened or was measured:

| Trap | What could go wrong | Guard |
|---|---|---|
| The id gives away the answer | corpus ids end in `_0` / `_1`, which reveals the label | sheets use neutral ids (`test_0001`); the link back is in a private mapping file annotators never open |
| Pair partners side by side | you judge by comparing the two answers instead of judging each one | rows are shuffled so partners are never adjacent |
| Pre-filled answers on test | the `type` column is `none` exactly when the answer is correct — pre-filling it would reveal the label | **test is never pre-filled** |
| Rubber-stamping pre-filled rows | on dev/train, an annotator could just click through | 10% of pre-filled rows are left **blank** as an attention check |
| Excel damage | Excel drops a leading `'`, rewrites Bengali digits, and runs `-1/4` as a formula | comparisons ignore those changes; formula-like cells are protected |
| Overwriting finished work | rebuilding the sheets once destroyed partial work | the builder refuses to overwrite filled sheets |

dev and train sheets arrived **90% pre-filled** (the corpus label plus a type where a simple rule
could decide it), to save time. The annotator **verified and corrected** them.

### 13.3 The hallucination types

For every **wrong** answer, the annotator picks a type:

| Condition | Type | Meaning | Example |
|---|---|---|---|
| has-context | `entity` | wrong person, place or organisation | passage says জিম্বাবুয়ে, answer says নিউজিল্যান্ড |
| has-context | `numeric` | wrong number, date or quantity | ১৭৭৪ vs ১৭৭৫ |
| has-context | `relational` | right things, wrong relationship | who seized vs who was detained swapped |
| has-context | `contradiction` | says the opposite of the passage | বিজারক vs জারক |
| no-context | `fabricated` | simply not true | — |
| no-context | `overclaim` | confidently answers something that has no fixed answer | — |
| (correct answers) | `none` | — | — |

**Why types?** A single score hides *what kind* of mistake a model misses. Types let us later report
"the model catches wrong numbers but misses scrambled relationships". Types are for **reporting
only** — a model never sees them, because they reveal the label.

### 13.4 Results

**All four sets are complete: 5,310 rows.** A validator
([`src/validate_annotation.py`](../src/validate_annotation.py)) checked every row:

| Check | Result |
|---|---|
| Structural errors (invalid label, type in wrong column, type not allowed for the condition) | **0** |
| Question/answer text changed or copied from the wrong row | **0** of 5,310 |
| Same item appearing in two splits | **0** |
| `unsure` / `unreadable` rows without an explanation | **0** of 191 |
| Pre-filled rows actually reviewed? | yes — **23%** (dev) and **26%** (train) were corrected |

**Test split agreement** ([`src/score_test_agreement.py`](../src/score_test_agreement.py)):

| Measure | Value |
|---|---|
| Items both annotators could label | 1,270 of 1,356 (86 marked `unsure`/`unreadable` by at least one) |
| Raw agreement | 1,184 / 1,270 = **93.2%** |
| **Cohen's κ** | **0.865** — "almost perfect", higher than the 0.717 pilot |
| Tawhid vs corpus label | 91.1% match |
| Shejan vs corpus label | 96.9% match |
| Agreement: has-context / no-context | 90% / 97% |
| Agreement: easy / hard | 95% / 88% |

Hard items agree less than easy ones — the expected direction. The rise from 0.717 to 0.865 shows
the new rules 11–15 worked.

**Is the annotation genuine?** Yes, and it can be shown: the two test annotators' sheets share 0
identical full rows and 0 identical notes despite 93% label agreement, and a quarter of pre-filled
dev/train rows were corrected rather than accepted.

> **Transparency note.** An LLM-assisted helper (`src/llm_annotate.py`) was written but **never
> run**. Every label in `data/annotated/round1/` was entered by a human.

---

## 14. Step 10 — Merging the annotation into the corpus

### 14.1 The problem this step solves

After step 9 the human judgements live in **spreadsheets**. The corpus files that models will read
still say `hallucination_type = unlabeled` everywhere. The merge copies the human results into
`corpus.jsonl` and the three split files.

(An earlier merge script could only read the output of the unused LLM helper and had no way to
read these spreadsheets. It was rewritten on 17 September 2026.)

**Script:** [`src/merge_annotation.py`](../src/merge_annotation.py)

```bash
python src/merge_annotation.py --dry-run     # show what would happen; change nothing
python src/merge_annotation.py               # write it (refuses while decisions are missing)
python src/audit.py --data data/splits       # always re-check the gate afterwards
```

### 14.2 The rules it follows

**Two things it will never change:**

1. **`label`.** If a human disagrees with the corpus label, the disagreement is **recorded** in
   `label_disputes.csv` — the label is *not* edited. The label is what every model is graded
   against; quietly changing it from a spreadsheet would change the exam after it was set. A
   genuinely wrong label must be fixed at its source and the corpus rebuilt.
2. **`difficulty`.** The corpus defines difficulty by a *rule* (§9). The guidelines asked annotators
   for their *feeling* of difficulty ("is it a believable near-miss?"). Those are different
   measurements, and they disagree on about 40% of rows — which is expected, not an error.

**How a type is decided:**

| Record | Result | `type_source` written |
|---|---|---|
| correct answer (label 1) | `none` | `definitional` |
| test, wrong answer, **both** annotators said wrong **and** chose the **same** type | that type | `double_annotated` |
| dev/train, wrong answer, the annotator said wrong and chose a type | that type | `single_annotated` |
| anything else (types differ, someone said correct, someone was unsure) | goes to **adjudication** | `awaiting_adjudication` → `adjudicated` once decided |
| train record outside the 20% sample | `unlabeled` (not required — §14.5) | `outside_train_sample` |

It also fills `annotator_1` / `annotator_2` (each human's 1/0, or empty for unsure) and
`adjudicated`.

**Adjudication** means a third look to settle a disagreement. The script writes
`data/annotated/round1/adjudication.csv`; for each row a person writes a type, `skip`, or `dispute`
(= "I think the corpus label itself is wrong"). Decisions are kept even if the script is rerun.

### 14.3 What the dry run shows (17 September 2026)

| Split | Wrong answers that need a type | Type settled by humans | Waiting for adjudication | Out of scope (§14.5) |
|---|---:|---:|---:|---:|
| test | 678 | 487 (71.8%) | 191 | — |
| dev | 673 | 648 (96.3%) | 25 | — |
| train (20% sample) | 626 | 589 (94.1%) | 37 | 2,503 |
| **total** | **1,977** | **1,724 (87.2%)** | **253** | **2,503** |

Why test needs the most adjudication: two people must agree on *both* the label *and* the type.
Of the 191: 81 have different types (mostly `relational` vs `entity`), 49 are correct-vs-wrong
splits, 45 involve `unsure`, and in 16 both annotators said the "wrong" answer is actually correct.

**Types so far** (the 1,724 settled):

| Condition | Distribution |
|---|---|
| has-context | numeric 436 · entity 218 · contradiction 167 · relational 132 |
| no-context | fabricated 771 · **overclaim 0** |

`overclaim` was never chosen. Our no-context questions all have a fixed, factual answer, so this is
a finding about the dataset, not a mistake.

**Label disputes:** **222 records** where at least one human disagrees with the corpus label;
in **116** of them every annotator who looked disagrees. Most often the corpus calls an answer
correct and the humans call it wrong (79 of the 116).

**Is the training data still trustworthy?** The plan's rule ([guide §5.3](IMPLEMENTATION_GUIDE.md))
says: *if the 20% sample shows more than 5% label noise, verify more.* After adjudication, the
train sample shows **4.5%** confirmed wrong labels (56 of 1,252 records). So, by our own pre-set
rule, the train labels are sound — though close enough to the limit to report plainly. Some errors do remain in the unchecked 80% — for example
this train pair nobody sampled:

> **Question:** চিত্তভূষণ দাশগুপ্ত কবে জন্মগ্রহণ করেন? (passage gives *৬ জুন ১৯১৫*)
> Marked correct: "২০১৪ সালে তার জন্মের শতবর্ষ পালিত হয়।" · Marked wrong: "৬ জুন ১৯১৫"
> The labels are swapped. Errors like this are why we measure the noise rate instead of assuming
> zero.

### 14.4 How the merge was verified

The write path was tested on a **copy** of the data, never the real files:

| Test | Result |
|---|---|
| Refuses to write while adjudication rows are empty | ✅ |
| Rejects an invalid decision (a has-context type on a no-context item) | ✅ |
| Keeps an adjudicator's decision and notes across reruns | ✅ |
| Running twice gives byte-identical files | ✅ |
| Labels changed | **0** of 8,960 |
| Difficulty changed | **0** of 8,960 |
| Split files still identical to the corpus rows | ✅ |
| `label 1 ⇔ type none` rule | 0 violations |
| Shortcut gate after merging | still **0.534 PASS** |

### 14.5 What is left in this step

**Done on 17 September 2026.**

| Item | Result |
|---|---|
| Adjudication | all 253 rows decided by hand: **143** given a type, **89** `dispute` (the stored "wrong" answer is actually correct), **21** `skip` (broken item) |
| Review of the decisions | 3 rows moved from a type to `skip`: `test_0761`, `dev_0285`, `dev_0692` differ from their partner answer only by spelling, and neither answer is in the passage (Rule 12). Original decision kept in the notes |
| Merge | written into `corpus.jsonl` and all three splits; `D8 MET` (1,867 of 1,977 in-scope wrong answers typed, 110 excluded) |
| Audit | strict mode: all structural checks pass; metadata probe still **0.534** |
| Final type counts | has-context: numeric 475 · entity 264 · contradiction 212 · relational 137 — no-context: fabricated 779 |

**What adjudication taught us.** Most disputes were one of two patterns, both already described in
the guidelines: a *typo-pair* (the "wrong" answer is the correct one with a spelling slip, Rule 12),
or a *sentence dump* (the "correct" answer is a whole passage sentence and the "wrong" one is the
exact fact, so both are right, Rule 14). Some pairs had their labels simply swapped — e.g. the
passage says "তাঁর বাবা ডা. ফখরুল আমিন খান", yet that answer was stored as wrong.

**Decided (17 September 2026): types are needed only where a human looked.**

The requirement D8 originally said *every* wrong answer in the corpus (4,480) needs a type. We
narrowed it to **test + dev + the 20% train sample (1,977)**. The other 2,503 train wrong answers
keep `hallucination_type = unlabeled`, marked `type_source = outside_train_sample`, so it is
visible that this is by design and not forgotten.

Why this is the right call, in plain words:

| Question | Answer |
|---|---|
| Do models use the type? | **No.** The type gives away the label (`none` = correct), so it can never be a model input. |
| Where is the type used? | Only in result tables that say "the model misses *numeric* errors more than *entity* errors". Those tables are computed on **dev and test**, which are fully typed. |
| Are the train *labels* affected? | **No.** This is only about the type. The train labels were checked on a 20% sample and have 4.5% confirmed noise — under our pre-set 5% limit. |
| What would typing the rest cost? | About 2,500 more annotations that would not change a single reported number. |

Full wording of the decision: [PRD §5.1c](PRD.md).

### 14.6 Leaving out pairs whose labels cannot be trusted

**What.** Human review found **180 pairs** where the stored labels are confirmed wrong or the item
is broken. These pairs are left out of **both training and scoring**.

| Why a pair is left out | train | dev | test |
|---|---:|---:|---:|
| the "wrong" answer is actually correct (adjudicator: `dispute`) | 28 | 19 | 42 |
| the "correct" answer was judged wrong by every annotator who saw it | 28 | 34 | 17 |
| the item is broken (adjudicator: `skip`) | 8 | 5 | 8 |
| **pairs left out** (a pair can have two reasons) | **62** | **54** | **64** |

**Why.**

- **Scoring:** if a model says "correct" to an answer that really is correct, but the stored label
  says "wrong", the model loses a point for being right. The score would then measure the dataset's
  mistakes, not the model.
- **Training:** the model would be taught the mistake.

**Why it is fair, not cherry-picking.**

1. The rule uses only **human** judgements, made before any model existed.
2. It was decided **before any model was trained or scored**, so it cannot have been chosen to
   make a result look better.
3. A **whole pair** is removed, never half of one, so every split stays exactly 50/50.
4. Nothing is deleted. The records stay in the files, marked `excluded = true` with a reason, so
   anyone can check or reverse the decision.

**How it is applied.** One small loader, [`src/splits.py`](../src/splits.py), is the only way
training and evaluation scripts read data, and it drops excluded pairs automatically:

```python
from splits import load_split
train = load_split("train")      # excluded pairs already removed
```

**What it changed.**

- **The gate still passes on the data models actually use:** metadata probe 0.514 (all records:
  0.534).
- **The string-matcher reference numbers were re-measured on the usable data:** 0.823 all · 0.987
  easy · 0.454 hard (§9.3).

**The dataset we will work with:**

| Split | Pairs | Records | has-context / no-context pairs | Hard has-context pairs |
|---|---:|---:|---|---:|
| **train** | **3,067** | **6,134** | 1,834 / 1,233 | 619 |
| **dev** | **619** | **1,238** | 353 / 266 | 109 |
| **test** | **614** | **1,228** | 359 / 255 | 119 |
| **total** | **4,300** | **8,600** | 2,546 / 1,754 | 847 |

Every split is still exactly 50% correct / 50% hallucinated.

---

## 15. Health check — was each step done correctly?

Each step was compared with what [PRD.md](PRD.md) and
[IMPLEMENTATION_GUIDE.md](IMPLEMENTATION_GUIDE.md) require, then re-measured.

| Step | Requirement | Verified | Verdict |
|---|---|---|---|
| 0 Scope | Bengali only; QA only; no RAG; no new data | no Banglish records; 0 fill-in-the-blank; no retrieval code exists | ✅ |
| 1 Foundations | frozen schema; `1 = correct`; fixed seeds; log | schema validated at build; no flip in any script; log has 12 rows | ✅ |
| 2 Audit | source audited before use | `DATASET_AUDIT.md` complete | ✅ |
| 3 Cleaning | duplicates removed | 62,084 → 56,480 reconciles exactly | ✅ |
| 4 Corpus | D1 ≥ 4,000 pairs · D2 60/40 · D3 50/50 · D6 no difficulty filter | 4,480 · 2,688/1,792 · exact · 13 subjects | ✅ |
| 5 Difficulty | D9 every pair easy/hard | 8,960 of 8,960 marked; 0.812 / 0.980 / 0.456 reproduced | ✅ |
| 6 Splits | D7 pairs never split; R6 deterministic | 0 shared pairs, 0 shared questions; byte-identical rebuild | ✅ |
| 7 Audit gate | V1 metadata probe < 0.60 | 0.534 | ✅ |
| 8 Pilot | D5 κ ≥ 0.60; R4 guidelines first | 0.717; guidelines predate annotation | ✅ |
| 9 Annotation | D4 test 100% double-annotated | 1,356 × 2, κ 0.865, 0 validation errors | ✅ (adjudication pending) |
| 10 Merge | D8 type for every wrong answer in test + dev + train sample (narrowed, §14.5) | 253 adjudicated; merged; 1,867 of 1,977 typed, 110 excluded; strict audit passes | ✅ |
| 10b Exclusion | Q5: flagged pairs out of training and scoring | 180 pairs marked `excluded`; loader filters them; gate 0.514 on usable data | ✅ |

### 15.1 Problems found and fixed on 17 September 2026

| # | Problem | Fix |
|---|---|---|
| 1 | `merge_annotation.py` could not read the human sheets at all | rewritten (§14) |
| 2 | `build_corpus.py` description claimed fill-in-the-blank and geography were *included* | corrected |
| 3 | `data/splits/README.md` listed wrong record counts (8,044 / 1,732 / 1,740) | → 6,258 / 1,346 / 1,356 |
| 4 | Guide said dev has 599 pairs | → 673 |
| 5 | PRD and guide said the hard subset has 1,600 records | → 1,854 (the 0.456 score was right) |
| 6 | Hard subjects quoted as 965 pairs / 21.5% | → 966 / 21.6% |
| 7 | "mathematics supplies 76% of the no-context pool" | → 59% (measured) |
| 8 | Guide said median passage length is 295 characters | → 266 in the corpus |
| 9 | `audit.py` warned "no Banglish records" — a leftover from the old Banglish-first plan | now checks for *non-Bengali* records instead |
| 10 | round1 README said 86 disagreements were 6.3% of 1,356 | → 6.8% of the 1,270 scored |
| 11 | test κ 0.865 was never written to the experiment log | added as `iaa_002` |
| 12 | Guide §5.3 still quoted planned sizes (500 / 500 / 3,000) | → actual sizes, plus the measured 3.3% noise rate |

### 15.2 Known issues still open (not errors in finished work, but must be resolved)

| Issue | Where | Must be settled by |
|---|---|---|
| Two licence questions (BCS source; licence of the released corpus). `LICENSE` says MIT, which likely cannot apply to CC BY-SA content | `data/SOURCES.md` | before the repo or corpus is made public |
| Which LLM built the original QA pairs is unrecorded (`generator_model = claude_version_unrecorded`); `level` (SSC/HSC/BCS) is `unknown` for all records | PRD Q4 | report honestly as a limitation |
| Guide suggests `max_length = 512`; PRD/cheat-sheet use `256` | guide §7.3 | decide at the start of step 11 |
| Guide's further-pretraining corpora are *Banglish* datasets (Phase 2 material) | guide §8.2 | choose unlabeled **Bengali** text at step 11 |
| Guide suggests 5-fold cross-validation on train+dev, while the splits are fixed | guide §7.4 | decide at step 11 |
| Guide's "human-written holdout" is impossible: every record is LLM-assisted | guide §9.2 | drop or redesign at step 11 |

---

## 16. All the parameters in one place

### 16.1 Data building

| Parameter | Value | Where | Why this value |
|---|---|---|---|
| Random seed | **42** (models: 42, 1337, 2024) | every script | reproducibility |
| has-context : no-context | **60 : 40** | `build_corpus.py` | main task is grounded; has-context data is scarce, so all of it is used |
| Label balance | **50 : 50** | by construction (pairs) | macro-F1 and κ behave cleanly on balanced data |
| Max subject share | **50%** has-context, **30%** no-context | `build_corpus.py` | stop mathematics (and similar) from dominating |
| Split | **70 / 15 / 15** train/dev/test, by pair, per (condition, subject) | `build_corpus.py` | enough to learn from; dev and test big enough to measure (≈ 1,350 records each) |
| Copied-question filter | **≥ 6** consecutive words from the passage | `build_corpus.py` | a question that contains its own answer is invalid |
| no-context "near-miss" (hard) | character similarity **≥ 0.60** | `build_corpus.py` | wrong answer close enough to be believable |
| Fill-in-the-blank detector | `শূন্যস্থান`, `___`, `....` | `build_corpus.py` | QA only |

### 16.2 Audit and gates

| Parameter | Value | Why |
|---|---|---|
| Metadata probe gate | **< 0.60** (< 0.55 = clean) | above 0.60 the answer's shape alone predicts the label |
| Probe train/test split | 80 / 20, grouped by pair | same leakage protection as the real splits |
| κ gate | **≥ 0.60** | "substantial" agreement; below it the guidelines need work |
| Leakage alarm | any macro-F1 **> 0.95** | too good to be true on this task |
| Label-noise rule for train | verify more if **> 5%** | measured: 4.5% after adjudication |

### 16.3 Annotation

| Parameter | Value | Why |
|---|---|---|
| Pilot size | **100** items (60 has-context / 40 no-context, 50/50) | enough to estimate κ, cheap enough to repeat |
| Rows per sheet | **250** | small enough to open, finish and save safely |
| Train sample | **20%** (626 pairs) | enough to measure noise without labelling everything |
| Pre-filled share (dev/train) | **90%**, with **10%** blank attention checks | saves time; blanks detect click-through |
| Test pre-fill | **none** | pre-filling the type reveals the label |

### 16.4 Model training (planned — step 11, not used yet)

**Lab-based models (Labs 1–5)**

| Parameter | Value | Plain meaning |
|---|---|---|
| Word n-grams / character n-grams | 1–2 words / 3–5 characters | pieces of text the sparse models count; character pieces survive Bengali word endings |
| Naive Bayes smoothing | α = 1 (Laplace) | add 1 to every count so an unseen word never gives probability 0 |
| Character n-gram LM order | n = 3 (2–4 tried on dev) | predict each character from the 2 before it |
| Skip-gram | 200 dimensions, window 5, words seen ≥ 2 times | each word becomes 200 numbers learned from its neighbours |
| Stop words / stemming | off by default; tested as variants | kept off because they can delete না / নয় and hide contradiction errors |
| RNN / BiRNN / BiLSTM hidden size | 256 (BiLSTM: 2 layers, dropout 0.3) | size of the network's running memory |
| RNN learning rate | 1e-3 (Adam) | larger than for BERT, because these start from almost nothing |
| Transformer from scratch | 128-number vectors, 4 attention heads, 2 layers | a small version of BERT's design, with no pretraining |
| Transformer learning rate | 5e-4 with 10% warm-up | from-scratch Transformers are unstable with bigger steps |
| Early stopping (RNNs, Transformer) | dev loss, patience 3 | stop when practice-test loss stops improving |
| Output | 1 number → sigmoid | the probability the answer is **correct** |

**Pretrained encoders (BanglaBERT etc.)**

| Parameter | Value | Plain meaning |
|---|---|---|
| Learning rate | 2e-5 | how big each learning step is; small, because the model is already pretrained |
| Optimizer | AdamW | the standard update method for transformers |
| Max length | 256 tokens | longest input kept; long passages are cut (cut the passage, never the question/answer) |
| Epochs | 5 | passes over the training data |
| Batch size | 32 (16 if memory runs out) | examples processed together |
| Warm-up | 10% | start with tiny steps, then speed up |
| Weight decay | 0.01 | mild penalty that discourages over-fitting |
| Early stopping | stop if dev loss does not improve for 2 epochs | avoids memorising train |
| fp16 | on | half-precision numbers — faster, less GPU memory |

---

## 17. Questions a teacher is likely to ask

**"Your has-context score is 0.82. Isn't that good?"**
No. A string matcher with no learning gets 0.823 on our usable data. That is why we report the
**hard subset** (string matcher 0.454) next to every has-context number, and why our target is 0.80 *on the hard
subset*.

**"Why is the test set used only once?"**
Every time a result on the test set influences a decision, the test set is partly "learned" by us,
and the final score stops being an honest estimate. We tune on dev as much as we like, and grade on
test once.

**"Why not split records randomly?"**
Then a question's correct answer could be in train and its wrong answer in test, and the model
could win by remembering questions. We split by pair; 0 pairs and 0 questions are shared across
splits.

**"The data was built with LLM help. Isn't that circular?"**
It is a real concern, which is why (a) humans independently checked all of test and dev and a 20%
sample of train, (b) we report how often humans confirm a stored label is wrong (4.5% in the
samples), and (c) we never let an LLM write labels.

**"Why keep law and science if they are so hard?"**
Removing them would make the benchmark easier than the real world. We keep them and report results
per subject, so any weakness is visible instead of hidden.

**"Why did κ go up from 0.717 to 0.865?"**
After the pilot we turned every disagreement into a written rule (rules 11–15). Clearer rules,
fewer disagreements.

**"Why are `unsure` items excluded from κ?"**
`unsure` means *the item itself is broken* (e.g. the passage does not answer the question), not
"I am not confident". An item nobody can label honestly is not a disagreement between people.

**"Why do you keep 'wrong' labels that humans dispute?"**
Because changing the target from a spreadsheet would silently change the exam. Disputes are listed
(222, of which 116 unanimous) and reported as a finding; any fix happens at the source followed by
a rebuild and a fresh audit.

**"Why don't all training records have a hallucination type?"**
The type is only used to break down results ("which kind of error does the model miss?"), and
those breakdowns are done on dev and test, which are fully typed. Models never see the type — it
would reveal the answer. So typing the remaining 2,503 train records would cost ~2,500
annotations and change no reported number. Their correct/wrong *labels* are still there and were
checked on a 20% sample (4.5% confirmed noise). This scope decision is written down in PRD §5.1c.

**"What does the metadata probe prove?"**
That answer length, digits and punctuation cannot predict correctness (0.534, close to a coin
flip). A model that scores well must be using the content.

**"Why no RAG?"**
With retrieval, a model could look up the fact instead of judging the answer. We would be measuring
the search step, and the closed-book condition would stop being closed-book.

**"How does this project use what you learned in the NLP lab?"**
Every rung of the model ladder comes from a lab (table in §18). Of the 33 topics taught, 26 are
used and 7 are deliberately left out, each with a written reason (PRD §5.3b). The lab code is
adapted, not copied: it was written for English, and run unchanged on Bengali it would delete every
Bengali letter and digit.

**"Why didn't you use lemmatization, spelling correction or the Seq2Seq model?"**
- **Lemmatization** needs a dictionary and part-of-speech tags that don't exist reliably for
  Bengali; the stemming test covers the same idea.
- **Spelling correction** would "fix" exactly the mistakes the detector must find.
- **Seq2Seq, text generation and sampling** produce text, and this project only judges text.
- **POS tagging** needs per-word labels we don't have.

**"Why not remove stop words, as the lab does?"**
Bengali stop-word lists contain না, নয় and নি. Removing them turns "বিভক্ত নয়" (not divided) into
"বিভক্ত" (divided) — the opposite meaning — and hides 212 contradiction errors. We keep them by
default, and run one variant *with* them removed just to measure the damage.

**"Why build a Transformer from scratch when BanglaBERT exists?"**
It has the same design as BanglaBERT but learns only from our ~6,000 records. The difference
between the two tells us how much of BanglaBERT's score comes from its pretraining on huge amounts
of Bengali text rather than from its architecture.

---

## 18. What comes next

### 18.1 How the project uses the NLP lab

The model ladder climbs through the lab syllabus in order, then goes one step beyond it. Every rung
is scored on dev; trained neural models run on 3 seeds.

| Lab | What the lab teaches | What we build with it | What it tells us |
|---|---|---|---|
| **1** | Regex cleaning, tokenization, stop words, stemming, edit distance | Bengali cleaner and tokenizer (default). Stop words and stemming as **variants** (negation always kept). Edit distance between the answer and its own passage | whether classic cleaning helps Bengali; how close a wrong answer is to the passage |
| **2** | Bag of Words, TF-IDF, N-gram language model | Word/character n-gram features; a **character n-gram LM** that scores how passage-like an answer is | the simplest learned detectors, and a softer version of the string trick |
| **3** | Skip-gram, cosine similarity, Naive Bayes, Logistic Regression, averaged and TF-IDF-weighted embeddings, the "dog bit the man" limit | Naive Bayes + Logistic Regression + SVM; Skip-gram vectors averaged two ways; cosine similarity of answer and passage; a **word-order test** on every model | whether meaning (not just exact words) helps; proof of what bag-of-words models cannot see |
| **4** | PyTorch, pretrained embeddings, RNN, BiRNN, BiLSTM, attention | Vanilla RNN → BiRNN → BiLSTM → BiLSTM + attention, starting from our Skip-gram vectors | what reading in order, in both directions, and paying attention add; *where* the model looked |
| **5** | Transformer encoder from scratch, positional encoding, self-attention | A small Transformer trained only on our data | how much of BanglaBERT's score comes from pretraining |
| beyond | — | BanglaBERT, MuRIL, XLM-R, mBERT, IndicBERT v2; further pretraining; ensemble; LLM reference | the strongest detector |

**Left out on purpose:** lemmatization, spelling correction, text generation (Shannon game, LSTM
sampling), POS tagging, Seq2Seq translation, word analogies — reasons in §17 and PRD §5.3b.

**Two new reference numbers come with this.** Besides the exact string matcher (0.823 / 0.454 hard),
a **fuzzy string matcher** ("the answer *nearly* appears in the passage") is reported too, so a
model that only learned approximate string matching cannot pass as a real detector.

### 18.2 In the order they will happen

1. **Step 11 — the model ladder** (milestones M4–M5), rungs in the lab order of §18.1:
   - **Lab 1–3 rungs:** Bengali cleaning and tokenization, n-gram models with Naive Bayes /
     Logistic Regression / SVM, the character n-gram LM, Skip-gram models, similarity features,
     and the preprocessing variants.
   - **Lab 4–5 rungs:** vanilla RNN, BiRNN, BiLSTM, BiLSTM + attention, Transformer from scratch.
   - **Beyond the lab:** BanglaBERT, MuRIL, XLM-R, mBERT, IndicBERT v2; further pretraining of mBERT /
     XLM-R; ensemble of the best three; an LLM zero-shot reference point; three input formats.
   - **Word-order test** on every trained model.
   - Classical models are *expected* to land around 0.50–0.65, and the from-scratch Transformer well
     below BanglaBERT. Those are correct results, not bugs.
2. **Step 12 — final evaluation** on `test.jsonl`, **once**, reported by condition, easy/hard,
   subject and hallucination type, with statistical tests (McNemar, bootstrap confidence intervals)
   before claiming any model beats another.

When each of these finishes, add its section here in the same What / Why / How / Result / Checked
format.

### 18.3 Step 11.1 (done) — Bengali text tools, `src/text_bn.py` (Lab 1)

**What.** One file that turns raw Bengali text into the list of words a model receives. Every model
that isn't a pretrained BERT uses it, through one call: `preprocess(text, variant)`.

**How — four cleaning steps, then splitting into words:**

| Step | Example | Why |
|---|---|---|
| 1. One spelling per letter (Unicode NFC) | য় stored two ways → one way | otherwise the same word can fail to match |
| 2. Remove footnote marks | `হয়।[1][২]` → `হয়।` | Wikipedia leftovers, 3,598 in train+dev |
| 3. One style of digit | `১৭০৪` → `1704` | same value, same word |
| 4. Tidy spaces | newlines, double spaces → one space | 208 passages had them |
| Split into words | `স্বাধীন হয়।` → `স্বাধীন` `হয়` `।` | the দাঁড়ি becomes its own token |

**The two Lab 1 ideas that needed care in Bengali:**

- **Stop words.** We use a published list (stopwords-iso, 398 words), copied unchanged. It contains
  **না, নয়** (not) and **একটি, দুটি, প্রথম** (one, two, first). Removing those would make
  "divided" and "not divided" look the same, and "one" and "two" look the same — exactly the errors
  we're detecting. So a second file, `configs/bn_protected_words.txt`, lists words the code never
  removes. Without that protection, 5,033 such words would vanish from train+dev.
- **Stemming** (cutting endings: কলেজের → কলেজ). A plain rule broke words: "শব্দের" (of the word)
  was cut into the non-word "শব্". The fix is one easy rule: **only cut if what's left is a real word
  seen in the training text.** Words ending in না/নি (not) are never cut.

**Why stop words and stemming are *off* by default.** Even the protected version removes some useful
words, such as "শুরু" (start). Whether they help or hurt is measured in the M10 experiment, not
assumed.

**Checked.** `python src/text_bn.py --check` runs 9 rules over all 19,118 train+dev texts — no digit,
Bengali letter, LaTeX `[7]`, negation or number word is ever lost. All pass. `python -m pytest tests/`
runs 27 small tests, one per rule. The test split was not read.

**To change something** (e.g. a teacher asks to keep "শুরু" too): add the word to
`configs/bn_protected_words.txt`, then rerun the two commands above.

### 18.4 Step 11.2 (done) — Comparing the answer with the passage, `src/features.py`

**The problem.** Counting words tells a model *which* words an answer uses. It does not tell it
whether those words match the passage sitting right next to them. So we measure that directly.

**What it produces.** For every record, a few simple numbers:

| Number | The question it answers |
|---|---|
| Found in passage? | Do these exact words appear in the passage? |
| How far off? | How many letters would have to change to find the answer in the passage? |
| Word overlap | What share of the answer's words are in the passage? |
| Sounds like the passage? | Would this passage naturally produce these letters? |
| Answer length, digit share | How long is the answer; how much of it is numbers |

**The two lab ideas behind it**

- **Edit distance (Lab 1)** — the number of letter changes between two pieces of text.
  "১৭০৪" is in the passage → 0 changes. "১৭০৫" → 1 change. Small number, close match.
- **N-gram language model (Lab 2)** — build a tiny model from *that record's own passage*, then
  ask how likely the answer's letters are. It's a gentler version of exact matching: it still
  gives credit when one letter is different.

**Two problems the real data caused, and the fixes**

- Passages run to 3,127 letters while answers average 16. Checking every position separately would
  take billions of steps, so we use the textbook "start anywhere" version of the edit-distance
  table. One pass, and it turned out to be *more* accurate too.
- 198 answers end with the Bengali full stop "।" where the passage doesn't. Before trimming it,
  the code reported those answers as missing from the passage when they were plainly there.

**What we found (the useful part).** The table shows how well each number separates right from
wrong answers. 0.50 means useless, 1.00 means perfect.

| Number | On records with a passage | **On the hard ones** |
|---|---|---|
| Found in passage? | 0.851 | 0.528 — almost useless |
| Word overlap | 0.774 | 0.613 |
| Sounds like the passage? | 0.700 | **0.657 — the best one** |

That is the whole reason this file exists. On hard records, exact matching dies — that is what
"hard" means — but the gentler measures still work.

**A tougher target for hard records.** We also tested a "nearly matches" rule: call an answer
correct if it *almost* appears in the passage.

| Rule | Score on hard dev records |
|---|---|
| Exact match | 0.487 |
| **Nearly matches** (about 40% of letters may differ) | **0.591** |

So a real model must beat **0.591** on hard records, not 0.487. Oddly, that same looseness *hurts*
everywhere else: on easy records exact matching is already nearly perfect, so allowing differences
only lets wrong answers slip through.

**One reassuring result.** We trained a model on the answer text *alone* — no question, no passage —
to check whether wrong answers have a give-away writing style. It scored 0.529–0.546, no better
than the existing answer-only check. Good: there is no fingerprint for a model to cheat with.

**Checked.** The fast edit distance is verified against a trusted library and against brute force.
24 tests. The test split was never opened.

### 18.5 Step 11.3 (done) — Deciding what the model reads, `src/preprocess.py`

**The problem.** A model has to be handed one piece of text. Which pieces, in what order? We build
three versions so we can test which works best:

| Version | What goes in |
|---|---|
| **F1** | question + answer |
| **F2** | passage + question + answer, all together |
| **F3** | passage as one part, question + answer as the other — like asking, "does this passage support this?" |

Records with no passage always come out as F1; there is nothing else to include.

**The one rule that matters.** A model can only read so much at once (our limit: 256 words and
punctuation marks). When a record is too long, **only the passage is shortened — never the question
or the answer.** If the answer were cut off, the model would be grading something it cannot see.

**Why we cut the passage from the end, and nothing cleverer.** We measured what the simple cut
actually costs, using the records where the answer can be found inside its own passage:

| Limit | Records needing a cut | Records that lose their evidence |
|---|---|---|
| **256 (what we use)** | 1.5% | **7 out of 1,634 — about 1 in 230** |
| 128 | 20.4% | 117 out of 1,634 |

A cleverer version — keep the part of the passage that best matches the **question** — saves only 2
more of those 7. Not worth the extra machinery.

**One tempting idea we refused.** We could keep the part of the passage that best matches the
**answer**. That would save many more — and it would be cheating. It would quietly hand the model
the evidence for correct answers only, which is the same shortcut this whole project exists to
avoid. The reason is written into the file so nobody adds it later.

**Checked.** On all 7,372 train and dev records: the answer and the question always survive, the
passage is only ever shortened (never altered), and both records of a pair get the same question and
passage — so **only the answer differs**, which is the thing being judged. 22 tests.

### 18.6 Step 11.4 (done) — The marking scheme, `src/evaluate.py`

**The problem.** If every model brought its own way of scoring, no two scores could be compared. So
this was built **before** the first model.

**What it reports for every model**

- The main score, how many hallucinations it **caught**, how many it **missed**, and how many good
  answers it **wrongly flagged**.
- The same numbers split by: with/without passage, easy/hard, school subject, and kind of mistake.
- The no-learning rules recomputed **on the same records**, with a plain verdict: does this model
  beat them or not?

**Three careful decisions**

| Decision | Why |
|---|---|
| The margin of error is worked out by resampling **whole question pairs**, not single answers | Both answers to one question share a passage, so they rise and fall together. Counting them separately would make a model look more certain than it really is. |
| For each *kind* of mistake, we report the **catch rate** instead of the main score | A group like "all the numeric errors" holds only wrong answers, so the usual score has no meaning there. |
| A separate row for **"has a passage + hard"** | That exact group is the project's 0.80 target. The plain "hard" row mixes in no-passage records and gives a different number (0.487 against 0.440). |

**Two gaps it prints instead of hiding**

- No "human-written versus AI-written" comparison, because every record in our corpus was built
  with AI help.
- Groups that are too small to trust are labelled "(small)". Some dev subjects have only 22
  records, where a score can swing wildly.

**Protection built in.** Scoring the test set refuses to run without an explicit `--final` flag,
because the test set may be used only once, right at the end.

**Proof it works.** It already scored the no-learning rules on dev: the exact-match rule gets 0.850
*with* a passage but 0.333 *without* one, and only 0.487 on the target group. 28 tests check the
arithmetic against answers worked out by hand, and the results match scikit-learn exactly.

### 18.7 Step 11.5 (done) — The first models that actually learn, `src/train_classical.py`

**What.** Twelve models, all trained on a laptop with no GPU, in seconds to minutes each. This is
the first point where the project produces real scores.

| Family | What it does | Lab |
|---|---|---|
| **M1** counting words | Counts which words appear, then learns which words go with wrong answers. Three learners: Naive Bayes, Logistic Regression, SVM | 2, 3 |
| **M11** language model | One model of how correct answers are written, one of how wrong ones are | 2 |
| **M2** word vectors | Learns a list of numbers per word from our own text, then averages them per answer | 3 |
| **M12** similarity numbers | Uses the measurements from step 11.2 (does the answer appear in the passage, how close is it) | 1, 2, 3 |

**The word vectors, built from scratch.** The library the guide named (`gensim`) has no version for
our Python, and Lab 3 teaches how to write Skip-gram by hand anyway — so we did. It reads our text
and learns which words keep similar company. It works:

| Word | What the model thinks is related |
|---|---|
| ১৯৭১ | মার্চ, জিয়াউর, ১৯৭২ |
| সরকার (government) | মুজিবনগর, প্রবাসী, গঠন |
| কবি (poet) | দাশ, শরৎচন্দ্র, সাহিত্যিক |
| ঢাকা | চট্টগ্রাম, বিশ্ববিদ্যালয় |

Nobody told it any of that — it worked it out from which words appear near each other.

**The results, and the one that matters most**

| Model | Score overall | Score on **hard** questions |
|---|---|---|
| Similarity numbers (M12) | **0.730 — the best** | **0.490 — nearly the worst** |
| TF-IDF + Logistic Regression | 0.549 | **0.610 — joint best** |
| TF-IDF + SVM | 0.548 | **0.610 — joint best** |
| Word vectors (M2) | 0.503–0.520 | 0.500–0.529 |
| *the two rules to beat* | | *0.487 and 0.591* |

**Read that top row carefully, because it is the whole lesson of this project.** The model with the
best overall score is the *least* useful one. Given the "does the answer appear in the passage?"
measurement, it leaned on it almost entirely — and on hard questions, where that trick stops
working, it fell to 0.490, below even the simple fuzzy rule.

Meanwhile TF-IDF + Logistic Regression looks mediocre overall (0.549) but, together with TF-IDF +
SVM, is the only kind of model that clearly beats both rules on hard questions (0.610 against 0.487
and 0.591). If we had judged by the overall number alone, we would have picked the wrong model.

Those two are an exact tie, and we checked rather than guessed: on the 218 hard questions they
disagree on 28, and each gets exactly 14 of them right. So we report a tie, which is what the
project's rules require when two models are this close.

**Two honest problems we report rather than hide**

- **Bag of Words + SVM never finishes settling.** Raw word counts here go up to 55 with no ceiling,
  and the SVM cannot converge on that even after 20,000 attempts. TF-IDF, which scales everything
  to at most 1, converges in seconds. Its score is therefore marked "unreliable" in the output and
  in the log. This is a neat demonstration of *why* TF-IDF's weighting is worth having.
- **The word vectors are the weakest family.** 345,000 words is a small amount of text to learn
  from — real word2vec uses billions — and averaging a sentence's vectors throws word order away.
  Running them on three different random starts changes the score by only ±0.002 to ±0.013, so this
  is a real limitation, not luck.

**Checking our code against the library's.** The guide asks us to prove the shortcut of using
ready-made libraries is safe. We wrote Naive Bayes out by hand the way Lab 3 does, and it predicts
**the same label on 100% of 200 dev records** as scikit-learn's version, with an identical score.

**Everything is trained on the train split only** — the vectorizers, the word vectors, the weights
and the models. Dev is only ever scored, never learned from. 35 tests cover this step.

---

### 18.8 Step 11.6 (done) — Does the textbook cleanup actually help? (M10)

**The question.** Every NLP course teaches the same three preparation steps: clean the text, throw
away very common words ("stop words"), and cut words down to their root ("stemming"). Lab 1 teaches
all three. But nobody in the lab ever checks whether they *help*. This step checks.

**How.** We take the two best models from step 11.5 and retrain each one six times, changing only
how the words were prepared:

| Name | What it does |
|---|---|
| **V0** | no cleaning at all — just split on spaces. The lower bound |
| **V1** | clean and tokenize properly. **This is the project's default** |
| **V2** | V1, then drop common words (negation words kept on purpose) |
| **V3** | V1, then stem |
| **V4** | V1, then drop common words *and* stem |
| **V2-demo** | V2, but negation and number words are dropped too — built to show the damage |

Everything else is held fixed: same model, same seed, same split, same input format. So any change
in the score comes from the word preparation and nothing else.

**Run it yourself:**

```bash
python src/train_classical.py --ablation --table --log     # about 25 minutes, CPU only
```

**The result** (full table: `results/tables/table6_preprocessing_ablation.csv`)

For TF-IDF + Logistic Regression:

| Variant | overall | hard | **contradiction** |
|---|---|---|---|
| V0 — no cleaning | 0.542 | 0.592 | 0.742 |
| **V1 — the default** | 0.549 | **0.610** | 0.712 |
| V2 — drop common words | 0.544 | 0.596 | 0.710 |
| V3 — stem | 0.551 | **0.610** | 0.742 |
| V4 — drop + stem | 0.549 | **0.610** | 0.726 |
| **V2-demo — negation dropped** | 0.530 | 0.609 | **0.528** ⚠️ |

**Three things this says, in plain words.**

**1. The textbook cleanup makes almost no difference.** Look at the first five rows: everything
lands between 0.542 and 0.551, and three of them have exactly the same hard score, 0.610. Stemming
(V3) is highest by 0.002 — but 0.002 on a single run is noise, not an improvement. The honest
answer to "does stop-word removal and stemming help?" is **no, not measurably.** We keep V1 as the
default and say so.

**2. Throwing away "না" is the one thing that really hurts.** V2-demo is the same as V2 except it
also deletes negation words (না, নয়, নেই) and number words (একটি, দুই, প্রথম). Its score on
`contradiction` errors falls from **0.712 to 0.528** — it loses about a quarter of its ability to
catch answers that say the opposite of the passage.

That makes sense the moment you say it out loud. A `contradiction` error is often a single word.
Here is a real pair from the dev split, not an invented example:

> **question:** প্রদত্ত তথ্য অনুযায়ী "পরিবারের একমাত্র সন্তান তিনি" সম্পর্কে কী জানা যায়?
> **correct answer:** পরিবারের একমাত্র সন্তান ছিলেন তিনি।
> **wrong answer:** পরিবারের একমাত্র সন্তান ছিলেন **না** তিনি।

The two answers differ by exactly one word. Delete "না" and they become *identical* — the model is
handed two copies of the same sentence and told one is right and one is wrong. There is nothing
left to learn from. This is why `configs/bn_protected_words.txt` exists, and this table is the
evidence that it is doing a real job rather than being a precaution nobody tested.

**3. The most important part: that damage is nearly invisible in the headline score.** Look at
V2-demo's `hard` column — **0.609**, against the default's 0.610. One thousandth apart. If we had
reported only the overall score or only the hard score, we would have concluded that deleting
negation words costs nothing at all. The harm only appears when the results are broken down by
error type.

This is the clearest example in the whole project of why §12.1 insists on slicing the results
instead of reporting one averaged number. A single number hid a 26% collapse.

**One more finding.** For the word-vector model, dropping common words is actively harmful: it
falls to **0.501** on hard questions, the worst cell in the table, and its best variant is V0 —
*no cleaning whatsoever*. Skip-gram learns a word's meaning from the words sitting around it, so
deleting the most frequent words punches holes in every piece of context it learns from.

**And a fair reason to use the cleanup anyway.** V4 needs 201,504 features where V0 needs 259,454 —
54,000 fewer columns for the same score. So the cleanup buys a smaller and faster model. That is a
real benefit; it is just not the benefit the textbook claims.

---

### 18.9 The demo interface — seeing the models make a decision

Everything up to here produces numbers in a table. This step makes the models usable: you
type a question and an answer, and all twelve say what they think of it.

**Start it:**

```bash
python src/serving.py --build     # once, about six minutes. Trains all 12 and saves them
python src/app.py                 # then open http://127.0.0.1:8000
```

#### What the page shows

The layout is two columns and fits on one laptop screen, so nothing needs scrolling. What you
type stays on the left; what comes back fills the right.

1. **The verdict count** — "8 of 12 models say this answer is correct". When the models
   disagree, that is not the demo malfunctioning; it is what a set of models scoring 0.49 to
   0.61 actually looks like.
2. **Every model, ranked** — ranked by its **score on hard questions**, not by how confident it
   sounds right now. Row 1 is the model with the best evidence behind it. A single prediction
   proves nothing; the scores are what say whether a model is any good.
3. **Two simple rules** — the exact and fuzzy string matchers, run on the same record. These
   involve no learning at all.
4. **The evidence** — does the answer appear in the passage, how many of its words are there,
   how many letters would have to change. These are the exact numbers the *Similarity numbers*
   model is given, not a separate explanation written for the page.

What you typed is not repeated back to you: it is still sitting in the form on the left.

Below that are eight real records from the dev split, each with both of its answers. Click
either one to load it into the form.

#### The thing worth showing your teacher

Try this pair, which is one of the clickable examples:

> **question:** ভারতীয় শাস্ত্রীয় সঙ্গীত অনুসারে মোট কতগুলি রাগ আছে?
> **passage says:** মোট **৬৫টি** রাগ আছে
> **answer given:** মোট **৬৭টি** রাগ আছে

The answer changes 65 into 67. It is a `numeric` hallucination, and the correct verdict is
*hallucinated*.

**Eight of the twelve models call it correct.** So does the fuzzy string matcher — only 2% of
the answer's letters have to change to find it in the passage, because the sentence is nearly
identical. The plain exact-match rule is the one that gets it right, for the crude reason that
"৬৭টি" does not appear anywhere in the passage.

That single screen is the argument for the entire rest of the project: these models compare
*shapes of text*. None of them can read a number out of a passage and check it against a
number in an answer. That is what the neural models and the pretrained encoders are for.

#### Why confidence is written three different ways

You will see "52% sure it is correct", "leans correct by 0.31", and "score gap +0.001". These
are not the same measurement dressed up differently:

- **Logistic Regression, Naive Bayes and XGBoost** give a genuine probability.
- **A linear SVM** only reports which side of its dividing line the record fell on, and how
  far. That is not a probability and has no upper limit.
- **The character language model** gives the gap between two likelihood scores.

Printing all three as a percentage would invent a certainty that two of them never expressed.

#### The scoreboard page

`/models` ranks every model by the hard-subset score, with the rules alongside. It is read
directly out of `results/experiment_log.csv` — the page cannot drift away from the results
table, because it has no numbers of its own. Models that ran on three seeds are shown as the
mean, exactly as Table 5 reports them.

Note the colour rule: green means the model **beat** the fuzzy rule (0.591). Bag of Words +
Logistic Regression lands on exactly 0.591, which is a tie, not a win — so it is grey. Only
two of the twelve are genuinely above that line.

#### One design rule behind all of this

The demo does not build its own copy of the feature pipeline. It calls `fit_model()` in
`src/train_classical.py` — the same function that produced the numbers in Table 5. If it had
its own copy, the two would drift apart the first time either changed, and the demo would be
showing predictions from a model nobody ever measured. There is a test that re-fits a model
from scratch and checks the saved copy agrees with it on 200 dev records.

The test split is not used anywhere in the demo. It trains on train, quotes dev scores, and
the clickable examples come from dev. There is a test for that too — it parses both files and
fails if either one so much as mentions loading `test`.
