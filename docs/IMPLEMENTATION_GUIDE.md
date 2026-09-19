# Phase 1 Implementation Guide — Bengali

**What this file is:** the *how-to* companion to [PRD.md](PRD.md) (the *what/why*). If the two ever disagree, PRD.md wins.

**Scope:** build a working Bengali hallucination detector, and test the full required model ladder. Bengali script in, Bengali script out.

**Not in this phase:** Banglish, transliteration, code-mixing studies, marking exact wrong words, or writing the final paper. Those are Phase 2 — see PRD.md §3.2.

**Version:** 2.2 — rewritten for clarity, 17 September 2026.

---

## 📍 STATUS AT A GLANCE

| Part of the pipeline | Status | Script |
|---|---|---|
| Clean the raw data | ✅ Done | `src/build_bn_pool.py` |
| Build the corpus + splits | ✅ Done | `src/build_corpus.py` |
| Human agreement test (pilot) | ✅ Done — score 0.717 | `src/build_agreement_test.py`, `src/score_agreement.py` |
| Full human annotation | ✅ Done — score 0.865 on test | `src/build_annotation_sheets.py`, `src/validate_annotation.py` |
| Merge annotation into the corpus | ✅ Done | `src/merge_annotation.py` |
| Safety/shortcut check | ✅ Passing | `src/audit.py` |
| Load data safely (skips 180 bad pairs) | ✅ Ready to use | `src/splits.py` |
| Bengali text cleanup tools | ✅ Done — 27 tests pass, `--check` passes on train+dev | `src/text_bn.py`, `tests/test_text_bn.py` |
| Hand-made features (n-gram LM, edit distance) | ✅ Done — 24 tests pass; V6 baseline measured | `src/features.py`, `tests/test_features.py` |
| Input-format builder | ✅ Done — 22 tests pass; answer/question never truncated | `src/preprocess.py`, `tests/test_preprocess.py` |
| Simple models (M1, M2, M11, M12) | ✅ Done — 35 tests pass; first real scores | `src/train_classical.py`, `src/skipgram.py` |
| Neural models (M3, M13) | ✅ Done — 65 tests pass; `--check` proves 3 fixes to the lab code | `src/train_neural.py`, `tests/test_train_neural.py` |
| Pretrained transformers (M4/M5) | ⬜ Not built yet | `src/train_transformer.py` (planned) |
| Further pretraining (M6) | ⬜ Not built yet | `src/further_pretrain.py` (planned) |
| Scoring + result tables | ✅ Done — 28 tests pass; Table 5 written | `src/evaluate.py`, `tests/test_evaluate.py` |

**In short: the data side of this guide (sections 1–5) is finished and just needs to be understood. The model side (sections 6–12) has started: steps 1–6 are done — Bengali text tools (§6), the hand-made features (§7.1a, §7.1b), the input formats (§6), the scoring code (§12), the first real models (§7.1, Table 5), and the M10 cleanup experiment (§6, Table 6). Next are the neural models.**

---

## 0. When is Phase 1 "done"?

All of these must be true:

| # | What must be true | Target | Status |
|---|---|---|---|
| C1 | Dataset built and annotated, agreement score reported | ≥ 4,000 pairs, agreement ≥ 0.60 | ✅ Done |
| C2 | Best model scores well on hard has-context items | ≥ 0.80 | ⬜ Not tested yet |
| C3 | Best model scores well on no-context items | ≥ 0.60 | ⬜ Not tested yet |
| C4 | The safety/shortcut check passes | Below 0.60 | ✅ Passing |
| C5 | Every has-context score is shown with its baseline | 0.823 all / 0.454 hard | ⬜ To do once models exist |
| C6 | A full log of every run exists | Every run logged | ✅ In place, growing |
| C7 | No model is allowed to search or look things up | Zero searching code anywhere | ✅ Checked, clean |
| C8 | Every usable NLP-lab topic is covered somewhere in the ladder | See PRD §5.3b | ⬜ To do as models are built |

**Two of these are non-negotiable:**
- **C4** — a high score that only passed because of a cheap trick is worse than no score at all.
- **C5** — a plain string-matching rule already scores 0.823 on has-context data. If you report "0.83" without saying that, the number is meaningless. Read §3.3 before writing up any result.

---

## PART 1 — THE DATA (this part is finished)

## 1. Environment setup

### 1.1 Hardware

A free Kaggle GPU (T4 ×2 or P100, 30 hours/week) or Google Colab is enough for everything in Phase 1. A normal-sized pretrained model fine-tunes on ~5,000 examples in 10–25 minutes. Further pretraining (§8) takes 2–4 hours.

**CPU-only note:** sections 1–6 of the model ladder below (the classical models, the n-gram model, the similarity features) run fine on a laptop CPU in minutes. Only the pretrained-transformer steps (§7.2, §8) genuinely need a GPU.

### 1.2 What to install

```bash
pip install -q transformers datasets accelerate evaluate
pip install -q scikit-learn pandas numpy scipy matplotlib seaborn
# gensim is NOT used: it has no build for Python 3.14, so Lab 3's Skip-gram is
# written out in src/skipgram.py with numpy only
pip install -q torch torchtext
pip install -q sentencepiece protobuf
pip install -q krippendorff        # human-agreement check
pip install -q sacremoses regex
pip install -q xgboost             # for M2
pip install -q rapidfuzz           # fast edit-distance, for M12
# statsmodels is NOT needed: the McNemar test is written out in src/evaluate.py

# Required specifically for BanglaBERT / BanglishBERT:
pip install -q git+https://github.com/csebuetnlp/normalizer
```

### 1.3 What the project folder actually looks like

`✅` = exists today. `⬜` = planned, not built yet.

```
Bhibranti/
├── NLP Lab/                        # the college lab guides + notebooks — read-only reference
├── configs/
│   ├── schema.json                 ✅ the fixed record format
│   ├── bn_stopwords.txt            ✅ published Bengali stop-word list, unchanged (see §6)
│   ├── bn_protected_words.txt      ✅ negation + number words that are never removed or stemmed
│   └── bn_suffixes.txt             ✅ Bengali stemmer endings (see §6)
├── data/
│   ├── raw/bn_qa_pool/             ✅ 14 original source files, untouched
│   ├── interim/bn_pool.jsonl       ✅ the cleaned-up pool
│   ├── corpus/bn_v1/corpus.jsonl   ✅ the final corpus
│   ├── splits/                     ✅ train / dev / test — always load via src/splits.py
│   └── annotated/                  ✅ the human labelling work (agreement_test_v1/, round1/)
├── src/
│   ├── build_bn_pool.py, build_corpus.py, build_agreement_test.py       ✅ builds the data
│   ├── build_annotation_sheets.py, validate_annotation.py               ✅ annotation tools
│   ├── score_agreement.py, score_test_agreement.py, merge_annotation.py ✅ scoring + merging
│   ├── audit.py                    ✅ the safety check
│   ├── splits.py                   ✅ the ONLY approved way to load data
│   ├── text_bn.py                  ✅ Bengali cleanup + tokenizer + stop words + stemmer
│   ├── features.py                 ✅ the n-gram language model + similarity features
│   ├── preprocess.py               ✅ builds the 3 input formats
│   ├── skipgram.py                 ✅ Lab 3's Skip-gram, written out (gensim has no 3.14 build)
│   ├── train_classical.py          ✅ M1, M2, M11, M12
│   ├── train_neural.py             ⬜ M3, M13
│   ├── train_transformer.py        ⬜ M4, M5
│   ├── further_pretrain.py         ⬜ M6
│   └── evaluate.py                 ✅ scoring, breakdowns, word-order test, significance tests
├── tests/
│   └── test_text_bn.py             ✅ one test per text rule — python -m pytest tests/
├── notebooks/
├── results/
│   └── experiment_log.csv          ✅ every run ever done, logged here
└── docs/
```

From day one, `results/experiment_log.csv` records **every** run — including the ones that failed — with this header:

```
run_id,date,model,input_format,preprocessing,seed,lr,batch,epochs,split,dev_macro_f1,test_macro_f1,notes
```

---

## 2. The record format

Every single record in the dataset looks like this:

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

### 2.1 The correct/wrong label — never changes, never flips

```
label = 1  →  CORRECT / FAITHFUL     (no hallucination)
label = 0  →  INCORRECT / HALLUCINATED
```

Read it as one simple yes/no question: **"is this answer correct?"** `1` = yes, `0` = no.

This is the *same* rule everywhere in the project — the schema, every script, every result table. It matches how the original source files already worked, so **no script anywhere flips this label**.

A few things follow from this:

- `1` is the *good* class here. Most other hallucination-research papers use the opposite convention (`1` = hallucinated). Macro-F1 doesn't care about this difference, but per-class precision/recall do — always say which class you mean.
- `hallucination_type` is `none` exactly when `label == 1`.
- If a model outputs two probabilities, the second one (`probs[:, 1]`) means "probability this is correct."
- So a detector's actual job is really to predict `0` (spot the hallucination).

Two fields are here for Phase 2 but cost nothing to fill in now: `script_condition` and `cmi` (Code-Mixing Index):

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

## 3. The corpus — how it was built (already done)

**Phase 1 generates no data.** The corpus already exists, built by a deterministic script. Read `src/build_corpus.py`'s own comments before changing anything about the data.

```
data/raw/bn_qa_pool/              14 original files, 62,084 records
  → src/build_bn_pool.py          cleans, removes duplicates, fits the fixed schema
data/interim/bn_pool.jsonl        56,480 cleaned records
  → src/build_corpus.py           pairs answers, filters, splits into train/dev/test
data/corpus/bn_v1/corpus.jsonl    4,480 pairs / 8,960 records
data/splits/{train,dev,test}      3,129 / 673 / 678 pairs  (usable after §3.4: 3,067 / 619 / 614)
  → src/merge_annotation.py       writes the human labels + excludes bad pairs
```

Both build scripts are deterministic: running them again with seed 42 gives byte-for-byte the same output. **Rebuilding the corpus resets every human label back to "unlabeled"** — so if you ever rebuild it, you must re-run `src/merge_annotation.py` afterward.

### 3.1 Every subject is included

Nothing was removed for being hard. Law, science, BCS, and literature together are 966 pairs (21.6% of the corpus) — same footing as everything else.

**The one exception:** fill-in-the-blank questions are excluded, because this project is Q&A only. A fill-in-the-blank question just teaches a model to copy words, and a dumb rule scores 0.929 on those alone. Removing them also removed the `geography` subject, since every geography item was fill-in-the-blank.

An earlier version of this script tried to filter out "hard to verify" items and dropped about 4,150 pairs by mistake. **That filter was removed.** Difficulty is now *measured* (the `difficulty` field), never filtered away (§3.3).

Two things to keep in mind when working with this:

1. Human checkers could look things up for no-passage items, or mark `unsure`. For has-passage items, they could never look anything up (see PRD §5.1b).
2. Always report scores broken down by subject. Law, science, BCS, and literature ask for facts a closed-book model just can't know. If a model does badly there, that's a real, reportable finding — not something to hide.

### 3.2 What actually gets filtered out, and why

Only four things are removed. None of them are about difficulty:

| Filter | Removes | Why |
|---|---|---|
| Incomplete pairs | A group missing either a correct or wrong answer | Can't form a pair without both |
| **Fill-in-the-blank** | Cloze-style questions (`___`, শূন্যস্থান পূরণ) | Q&A only — a dumb rule scores 0.929 on these |
| Damaged text | OCR scanning errors that leave broken characters | Unreadable, so unlabelable |
| Self-answering questions | A question that copies 6+ words straight from its own passage | The question gives away its own answer |
| Duplicate questions | The exact same question appearing twice | Would leak between train and test |

Plus two rules about the overall *mix* of the data (not about quality): keep the 60/40 has-context/no-context split, and cap how much any one subject can dominate — without a cap, mathematics alone would make up 59% of the no-context data.

`build_corpus.py` double-checks at the end of every build that zero fill-in-the-blank items got back in.

**One filter that was tried and then removed:** an early version dropped any pair where the *wrong* answer also happened to be a word-for-word quote from the passage. That was a mistake — it left only the "easy" pairs where "quoted from the passage" and "is correct" always line up together, which artificially inflated the string-matching score. Those pairs are kept now, because they're exactly the ones a shortcut can't solve.

### 3.3 The two shortcuts we had to check for

**Shortcut 1 — surface features (the hard safety gate).** Could a model guess the label just from things like answer length, digit count, or punctuation, without understanding anything? Currently: **0.534** on all data, **0.514** on the cleaned-up data. Both must stay below 0.60. **Passing.**

**Shortcut 2 — exact word matching (must be reported, never hidden).** "If the answer's exact words appear in the passage, call it correct" scores **0.823** on all cleaned-up has-context items. This happens because correct answers in this kind of data are usually copied word-for-word from the passage, and wrong answers usually aren't.

The `difficulty` field marks where this trick *doesn't* work — those are the "hard" items, where a model actually has to read and understand.

| Group | Cleaned-up data (the reference number) | All data (before cleanup) |
|---|---:|---:|
| All has-context | **0.823** (5,092 items) | 0.812 (5,376 items) |
| Easy | 0.987 (3,398 items) | 0.980 (3,522 items) |
| **Hard** | **0.454** (1,694 items) | 0.456 (1,854 items) |

**Rule: never report a has-context score by itself.** Always show these baselines next to it. And when scoring dev or test specifically, recompute the baseline on that exact same set of items — don't reuse the whole-corpus number.

### 3.4 How to load data — always through `src/splits.py`

Human review found **180 pairs** (train 62, dev 54, test 64) with labels that just can't be trusted. These are excluded from both training and scoring (see PRD §5.1d).

They're not deleted — both records are kept, just flagged `excluded = true`. One small script removes them for you:

```python
from splits import load_split          # run your script from the repo root
train = load_split("train")            # 3,067 pairs / 6,134 records
dev   = load_split("dev")              #   619 pairs / 1,238 records
test  = load_split("test")             #   614 pairs / 1,228 records — only touch this at M6
```

**Never read `data/splits/*.jsonl` directly.** Always go through `load_split()`. It also refuses to run if the human annotation hasn't been merged in yet — which protects you from accidentally training on a freshly-rebuilt corpus that lost its labels.

### 3.5 How to rebuild everything from scratch

```bash
python src/build_bn_pool.py     # only needed if the raw source files changed
python src/build_corpus.py      # rebuilds the filtered, split corpus
python src/merge_annotation.py  # puts the human labels + exclusion flags back — a rebuild wipes them
python src/audit.py --data data/splits    # must pass before you use the result
```

---

## 4. The corpus is finished — no more data collection

**Do not collect, scrape, generate, or create more data.** The corpus is locked at 4,480 pairs. This section exists so that decision is never quietly undone later.

If a future phase genuinely needs more grounded Bengali Q&A, the options — in order of preference — would be: (1) regenerate the roughly 2,400 has-context pairs whose wrong answer isn't drawn from the passage (this is what causes the exact-match shortcut); (2) bring in an outside dataset like TyDiQA-GoldP, BanglaRQA, or BEnQA, recording its licence *first*; or (3) rewrite the excluded fill-in-the-blank items into real questions.

**What was tried and failed:** generating wrong answers by swapping spans of text with hand-written rules. It produced truncated words, broken sentences, and answers of the wrong *type* entirely (e.g. a place name answering "what was his father's name?"). The AI-written wrong answers already in the dataset are much better — don't try to "improve" on them this way.

---

## 5. Annotation (already done — this section is now a reference)

### 5.1 The process we followed

- At least 2 human checkers, both native Bangla speakers.
- Write the labelling rules down *before* starting — we used BanTH's rulebook as a model to follow.
- Run a small **100-item test batch** first, measure agreement, fix any confusing rules, *then* label everything else.
- A senior reviewer resolves any disagreement.
- Checkers label two things: `label` (correct/wrong) and, for wrong answers, `hallucination_type`.

### 5.2 How we measured agreement between the two checkers

```python
from sklearn.metrics import cohen_kappa_score
kappa = cohen_kappa_score(df.annotator_1, df.annotator_2)
```

**The bar to clear: 0.60.** Below 0.40 means the *rules* are broken, not the checkers. For comparison, a similar published project (BanTH) reported 0.71 between checkers and 0.75 against an expert.

If the score comes out low, it's usually because "hallucination" wasn't clearly defined for tricky cases — partly-correct answers, technically-true-but-unsupported answers, hedged answers ("maybe", "possibly"). Write an explicit rule for each tricky case you find.

### 5.3 How much of each split got checked, and by whom

| Data | How it was checked | Size |
|---|---|---|
| Test | 100% by hand, by **two** people, disagreements resolved by a third look | 678 pairs / 1,356 records |
| Dev | 100% by hand, by one person | 673 pairs / 1,346 records |
| Train | Original label kept, plus a **20% sample** double-checked by hand | 626 of 3,129 pairs checked (1,252 of 6,258 records) |

**The rule we followed:** if that 20% train sample showed more than 5% wrong labels, we'd have needed to check more of train.

**What we actually found, after resolving disagreements:** 56 of the 1,252 checked train records (**4.5%**) had a label a human confirmed was wrong — 28 where the "correct" answer was actually wrong, and 28 where the "wrong" answer was actually correct. That's under the 5% line, so train did **not** need wider checking.

At the level of whole pairs, 62 of the 626 checked train pairs got flagged this way (also 54 of 673 in dev, and 64 of 678 in test). All of these flagged pairs are now excluded from training and scoring — see PRD §5.1d. The full list of disagreements is saved in `data/annotated/round1/label_disputes.csv`. **We never edit a label ourselves** — a bad label is reported and excluded, not silently fixed.

**Error-type labels follow a narrower rule (PRD §5.1c, decided 17 September 2026):** a type label is only required for wrong answers in test, dev, and the 20% train sample — 1,977 items total. The other 2,503 train items keep `hallucination_type = unlabeled`, marked `type_source = outside_train_sample` so it's clear this was on purpose.

**Important: never feed `hallucination_type`, `type_source`, `annotator_1`/`annotator_2`, or `adjudicated` into a model.** Every one of these gives away the answer.

---

## PART 2 — THE MODELS (this part is not built yet — start here)

## 6. Cleaning up the text, and choosing an input format

Phase 1 is Bengali script throughout, so there's no need for the "convert-between-scripts" step that Phase 2 will need. What Phase 1 *does* need is: cleaning the text, splitting it into words, normalising it, and choosing how to format the input. Format usually matters more than which model you pick.

The next four sub-sections adapt **Lab 1** to Bengali (PRD §5.3c). They feed every model that isn't a pretrained transformer (M1–M3, M11–M14).

> ⚠️ **Never copy Lab 1's code straight onto Bengali text.** The lab's regex (`re.sub(r'[^A-Za-z\s]', '', text)`) deletes every Bengali letter and every digit. NLTK's English tools (`word_tokenize`, `stopwords('english')`, `PorterStemmer`, `WordNetLemmatizer`) either do nothing or actively break Bengali. Use the Bengali versions below instead.

### Step 1: Cleaning (default for M1–M3, M11–M14)

Apply these steps, in this order, to the passage, question, and answer alike:

| Step | How | Why |
|---|---|---|
| Normalise Unicode | `unicodedata.normalize("NFC", text)` | The Bengali letter য় can be stored two different ways internally. Our corpus only uses one way (44,109 times) — but a stop-word list copied from elsewhere might use the other way and silently fail to match anything. Normalise the text *and* both word lists. |
| Remove citation marks | `\[[0-9০-৯]+\](?!\{)` | About 1 in 7 passages still has leftover Wikipedia footnote marks like `[1]`, `[2][3]` (3,598 in train+dev). The `(?!\{)` part keeps LaTeX roots such as `\sqrt[7]{n^3}`, whose `[7]` is always followed by `{`. |
| Make digits consistent | Map `০১২৩৪৫৬৭৮৯` → `0123456789` | So "১৯৭১" and "1971" are treated the same feature, without losing the actual number. |
| Collapse extra spaces | `\s+` → one space, then trim | 208 passages contain stray newlines or tabs. |

**Never delete:** digits (475 of our `numeric` labels depend on them), Bengali letters, negation words (see Step 3), or punctuation (turn it into its own token instead of deleting it). Also keep the invisible "joiner" characters used in some Bengali spellings.

### Step 2: Splitting into words (default)

```python
TOKEN = re.compile(r"[ঀ-৿‌‍]+|[0-9]+(?:[.,][0-9]+)*|[A-Za-z]+|[^\s]")
tokens = TOKEN.findall(clean(text))
```

This keeps: runs of Bengali letters, numbers (including decimals), English words, and every other non-space character (including the Bengali full stop, দাঁড়ি `।`) as its own token. It needs no external library, so every model downstream sees exactly the same tokens.

### Step 3: Removing common words — only as a test, never the default (M10)

- The list is the published **stopwords-iso** Bengali list (398 words, MIT), copied **unchanged** into `configs/bn_stopwords.txt` so anyone can check it against the source.
- That list contains negation words (না, নয়, নেই, নাই, ছাড়া) **and** number words (একটি, দুই, দুটি, চার, হাজার, প্রথম …). Removing negation turns "not divided" into "divided"; removing numbers hides errors like "একটি" → "দুটি".
- So these words live in `configs/bn_protected_words.txt`, and the code **never removes them**. To protect another word, add it to that file — don't edit the published list.
- The **V2-demo** variant uses the published list with no protection, to measure the damage. On train+dev it removes 5,033 protected words that V2 keeps.
- Even with protection, stop-word removal deletes some meaning-carrying words, e.g. "শুরু" (start) — in "…১৭০৪ সালে **শুরু** হয়" that word is the whole difference between the right and wrong year. This is why removal is an experiment, never the default.

### Step 4: Stemming — only as a test, never the default (M10)

Follow the same shape as the lab's Porter Stemmer, adapted for Bengali:

1. Try the endings in `configs/bn_suffixes.txt`, longest first, cutting at most one per word — noun endings only, e.g. `গুলোকে`, `গুলো`, `দের`, `েরা`, `রা`, `টির`, `টি`, `ের`, `এর`, `কে`, `তে`.
2. Only cut if **at least 3 characters** are left (`MIN_STEM_LENGTH`), so "ধারা" isn't cut to "ধা".
3. Only cut if what's left is a **real word seen at least twice in the train text** (`KNOWN_WORD_MIN_COUNT`). Without this, "শব্দের" would be cut at "দের" into the non-word "শব্"; with it, the cut is at "ের", giving "শব্দ". Only train is used, so dev/test can't influence it.
4. **Never** cut protected words, or words ending in a negation (`নি`, `না`) — "করেননি" (did not do) must never become "করেন".
5. Leave numbers and English words alone.

Not in the list, on purpose: single-character endings like "র" ("সরকার" would become "সরকা"), and verb endings (tense can *be* the error: "হয়েছে" vs "হবে"). On train+dev, stemming changes 6.8% of tokens (সালের → সাল, শব্দটির → শব্দ, নেতাদের → নেতা).

(We don't attempt full lemmatization — it needs a dictionary and grammar-tagging tool that doesn't reliably exist for Bengali. See PRD §5.3b.)

### The M10 experiment: does any of this actually help?

Run these variants on the best model from M1, M2, and M3, dev set only:

| Variant | Clean + split | Remove stop words (negation kept) | Stem |
|---|:-:|:-:|:-:|
| V0 (just split on spaces — the lower bound) | — | — | — |
| **V1 — the default** | ✅ | — | — |
| V2 | ✅ | ✅ | — |
| V3 | ✅ | — | ✅ |
| V4 | ✅ | ✅ | ✅ |
| V2-demo (also removes negation, to show the damage) | ✅ | ✅ (incl. negation) | — |

Report overall score, hard-item score, and the `contradiction` error-type score for each — that last one is the number to watch.

**All of this is built.** Every model gets its words through one call: `preprocess(text, variant="V1")` in `src/text_bn.py`.

#### Table 6 — the result (`python src/train_classical.py --ablation --table --log`)

Dev split, F2 input, seed 42. One model per family: `tfidf_logreg` (the joint-best M1) and `skipgram_mean_xgb` (the best M2). One seed rather than three, because this experiment is about the gap *between* variants, not the last thousandth of any one of them. Full table: `results/tables/table6_preprocessing_ablation.csv`.

| Model | Variant | overall | **hard** | **contradiction** | numeric | relational | features |
|---|---|---|---|---|---|---|---|
| `tfidf_logreg` | V0 — no cleaning | 0.542 | 0.592 | 0.742 | 0.558 | 0.586 | 259,454 |
| | **V1 — the default** | 0.549 | **0.610** | 0.712 | 0.561 | 0.581 | 215,601 |
| | V2 — drop common words | 0.544 | 0.596 | 0.710 | 0.548 | 0.581 | 209,650 |
| | V3 — stem | **0.551** | **0.610** | 0.742 | 0.573 | 0.571 | 207,385 |
| | V4 — drop + stem | 0.549 | **0.610** | 0.726 | 0.555 | 0.586 | 201,504 |
| | **V2-demo — negation dropped** | 0.530 | 0.609 | **0.528** ⚠️ | 0.549 | 0.570 | 209,054 |
| `skipgram_mean_xgb` | V0 — no cleaning | **0.522** | 0.557 | 0.558 | 0.549 | 0.541 | 90.1% covered |
| | V1 — the default | 0.519 | 0.557 | 0.513 | 0.539 | 0.493 | 94.0% |
| | V2 — drop common words | 0.498 | **0.501** ⚠️ | 0.528 | 0.509 | 0.440 | 92.7% |
| | V3 — stem | 0.515 | 0.523 | 0.552 | 0.533 | 0.542 | 94.5% |
| | V4 — drop + stem | **0.522** | 0.568 | 0.633 | 0.534 | 0.533 | 93.2% |
| | V2-demo — negation dropped | 0.518 | 0.557 | 0.491 | 0.513 | 0.531 | 92.6% |

**What it says.**

1. **The classic cleanup is worth almost nothing here.** Across V0–V4, `tfidf_logreg` moves between 0.542 and 0.551 overall and its hard score sits at 0.610 for three of the five. Stemming (V3) is nominally best at 0.551, but 0.002 above the default on one seed is not a result — it is noise, and the honest report is "no measurable difference". The textbook pipeline of stop-word removal and stemming, applied to this task, **does not earn its place.**

2. **V2-demo is the finding, and it is a large one.** Dropping negation and number words costs `tfidf_logreg` **0.184 on `contradiction`** (0.712 → 0.528), a 26% relative collapse. That is not noise — it is four times the entire spread of every other variant combined.

3. **And that damage is nearly invisible in the headline numbers.** V2-demo's overall score is 0.530 and its hard score 0.609 — the hard score is *within 0.001 of the default*. Anyone reading only the overall or hard column would conclude the negation words did not matter. The damage shows up in exactly one place: the error type that depends on the word "না". This is the strongest argument in the project for slicing results by error type rather than reporting one averaged number (§12.1).

4. **Stop-word removal actively hurts the word vectors.** `skipgram_mean_xgb` drops to 0.501 on hard under V2 — the worst cell in the table. Skip-gram learns from which words sit near which; deleting the most frequent words tears holes in every context window it trains on. Notably V0, with no cleaning at all, is its joint-best variant.

5. **Fewer features, same score.** V4 uses 54,000 fewer columns than V0 (201,504 vs 259,454) for an identical overall score. So the cleanup buys a smaller, faster model — just not a better one. That is a legitimate reason to use it, and not the reason the textbook gives.

**Decision: V1 stays the default**, unchanged. V3's 0.002 is not grounds for a switch, and V1 is the variant every earlier result was measured under.

```bash
python src/text_bn.py "কলেজের ছাত্ররা বইটি পড়েনি।"   # see every step on one sentence
python src/text_bn.py --demo                          # see every step on 5 examples
python src/text_bn.py --check                         # 9 safety rules on all train+dev text
python -m pytest tests/test_text_bn.py -v              # 27 tests, one per rule
```

Measured on train+dev (V1 → V4): 408,030 tokens and 21,556 distinct words with V1; stop-word removal cuts tokens to 330,602; stemming cuts distinct words to 19,338 (V3) and 19,050 (V4).

### Normalising text for BanglaBERT

BanglaBERT was trained on text passed through a specific normaliser tool. Skip this and you lose accuracy for free:

```python
from normalizer import normalize
text = normalize(text)
```

Use it consistently at both training and prediction time, and note in your log whether you used it. For mBERT, XLM-R, and MuRIL, it's optional — test it as an experiment rather than assuming it helps.

### Choosing an input format (M9)

| Format | What it looks like | Notes |
|---|---|---|
| F1 (plain) | `question + " " + answer` | The only option when there's no passage |
| F2 (with passage) | `passage + " " + question + " " + answer` | |
| F3 (entailment-style) | `[CLS] passage [SEP] question + answer [SEP]` | Usually works best when there's a passage |

F3 frames the task like "does this passage support this answer?" — close to what pretrained language-understanding models are already good at.

**Truncation matters a lot here.** If you cut the input at 256 tokens and the part of the passage that actually answers the question gets cut off, you're training on noise. **Always cut the passage, never the question or the answer.**

**Built and measured** (`src/preprocess.py`, checked on all 7,372 train+dev records):

| Budget | Has-context records needing a cut | Pairs whose evidence is lost |
|---|---|---|
| 512 tokens | 0.1% | — |
| **256 (the default)** | **1.5%** | **7 of 1,634 (0.4%)** |
| 128 tokens | 20.4% | 117 of 1,634 (7.2%) |

("Evidence lost" is measured on the pairs whose correct answer is a literal span of their passage, so its position can be located.)

So the passage is simply cut from the end, and nothing cleverer. Keeping the window that best matches the **question** was tried and rescues only 2 of those 7 — not worth the machinery. Keeping the window that best matches the **answer** would rescue more and is **forbidden**: it would hand the model the evidence for correct answers only, which is the string-matching shortcut all over again.

Typical finished sizes: F1 median 14 tokens, F2 median 34 (p99 246), F3 median 37.

```python
from preprocess import as_tokens, format_record
as_tokens(record, "F2")        # for the classical and recurrent models
format_record(record, "F3")    # ModelInput(text_a=passage, text_b=question + answer)
```

For a pretrained encoder, pass F3's two segments to the tokenizer with `truncation="only_first"` so it can only ever shorten the passage. Its sub-word tokenizer counts more tokens than ours does on Bengali, so for F1/F2 pass that tokenizer's length function into `record_parts(..., length_fn=...)` to pre-cut exactly.

---

## 7. The model ladder

### 7.0 No searching, ever (this rule can't be relaxed)

**No looking anything up at prediction time.** No vector database, no searching the training set for similar past examples, no fetching outside text. A model's input is only: the question, the passage already stored in that record (if there is one), and the candidate answer.

Searching would let a model just look up facts for law or science questions — which would measure how good its search is, not whether it can spot a hallucination. And it would make "no passage given" questions pointless.

Using word embeddings *as input features* is completely fine — Skip-gram vectors, or a pretrained encoder's internal representation, are just ways of representing text, not searching. The rule is only about *fetching other documents or examples* at prediction time.

Anything that isn't searching is fair game, and trying extra techniques is encouraged: n-grams, Skip-gram/word2vec, RNNs, LSTMs (with or without attention), BERT-style models, combining several models. You can add to the required ladder below — you can't remove anything from it.

### 7.1 The classical models — build these first

These are quick to build, give you a working pipeline in an afternoon, and their mistakes tell you useful things. The order below follows the lab: sparse word-counting models → the n-gram language model → word embeddings → similarity features → recurrent models → the from-scratch Transformer. Every model here uses the §6 cleanup and tokenizer, and loads its data with `load_split()`.

| ID | Model | How to build it | Lab | Also record |
|---|---|---|---|---|
| M1 | Bag of Words + Naive Bayes | `CountVectorizer`, word 1–2 grams + character 3–5 grams; `MultinomialNB(alpha=1.0)` | 2, 3 | — |
| M1 | TF-IDF + Naive Bayes | `TfidfVectorizer`, same n-grams; `MultinomialNB(alpha=1.0)` | 2, 3 | — |
| M1 | Bag of Words / TF-IDF + Logistic Regression | Same features, `LogisticRegression(max_iter=2000)` | 2, 3 | out-of-vocabulary rate on dev |
| M1 | Bag of Words / TF-IDF + SVM | Same features, `LinearSVC` | 2 | — |
| M11 | Character n-gram language model | See §7.1a below | 2, 3 | dev perplexity |
| M2 | Skip-gram (averaged) + Logistic Regression / XGBoost | gensim Word2Vec, `sg=1, vector_size=200, window=5, min_count=2`, trained **only** on the train split; average the word vectors | 3 | vocabulary coverage |
| M2 | Skip-gram (TF-IDF weighted) + Logistic Regression / XGBoost | Same vectors, weighted by TF×IDF | 3 | — |
| M12 | Similarity features + Logistic Regression | See §7.1b below | 1, 3 | which features mattered most |
| M3 | Plain RNN | 1 layer, 256 hidden units, one direction | 4 | — |
| M3 | Bidirectional RNN | 1 layer, 256 hidden units | 4 | — |
| M3 | Stacked BiLSTM | 2 layers, 256 hidden units, 30% dropout | 4 | — |
| M3 | BiLSTM + attention | Adds dot-product attention over the hidden states | 4 | attention weights on 20 dev items |
| M13 | Transformer built from scratch | See §7.1d below | 5 | — |

**Built and run** (`python src/train_classical.py --all --seeds`). Dev results, F2 input, V1 preprocessing. The M1 rows were re-measured on 2026-09-19 after a fix: each part's column had been picking up the `<SEP>` marker that belongs only in a flat sequence. It moved the scores by at most 0.014 - and not at all for TF-IDF, which gives a token appearing in every document almost no weight. The `hard` column is the PRD G4 target; `±` is the spread over three seeds where the model has randomness in it.

| Model | Lab | dev macro-F1 | has-context + **hard** |
|---|---|---|---|
| Skip-gram mean + XGBoost | 3 | 0.503 ±0.013 | 0.529 |
| Skip-gram TF-IDF + LogReg | 3 | 0.506 ±0.003 | 0.515 |
| Skip-gram mean + LogReg | 3 | 0.508 ±0.002 | 0.500 |
| Skip-gram TF-IDF + XGBoost | 3 | 0.520 ±0.007 | 0.518 |
| TF-IDF + Naive Bayes | 2, 3 | 0.527 | 0.592 |
| Character n-gram LM (M11) | 2 | 0.529 | 0.605 |
| BoW + Naive Bayes | 2, 3 | 0.529 | 0.596 |
| BoW + SVM | 2 | 0.542 ⚠️ | 0.559 |
| **TF-IDF + SVM** | 2 | 0.548 | **0.610 — joint best** |
| **TF-IDF + LogReg** | 2, 3 | 0.549 | **0.610 — joint best on the target** |
| BoW + LogReg | 2, 3 | 0.559 | 0.591 |
| **Similarity features (M12)** | 1, 2, 3 | **0.730 — best overall** | **0.490 — near worst** |
| *the bars to clear* | | | *exact 0.487 · fuzzy 0.591* |

**Four things this table says.**

1. **The best overall model is the least useful one.** M12 tops the table at 0.730, but its heaviest weight is `exact_in_passage (+1.95)`, it matches the string matcher *exactly* on has-context (0.850 = 0.850), and on hard it drops to 0.490 — below the fuzzy rule. It learned the shortcut, which is precisely what §3.3 warned about and why the hard column exists.
2. **Only TF-IDF + LogReg and TF-IDF + SVM clear both bars on hard** (0.487 exact, 0.591 fuzzy). They tie at **0.610**, and it is a real tie, not a rounding coincidence: on the 218 hard records they disagree on 28, and each is right on exactly 14 of them (McNemar p = 1.000). Reported as a tie, per §12.2.
3. **Skip-gram is the weakest family (0.50–0.52).** 345,000 words of training text is tiny for word vectors, and averaging them throws word order away. The seed spread is small (±0.002–0.013), so this is not noise.
4. **Everything lands in the 0.50–0.65 band §12.3 predicted.** No pretraining, no surprise.

**Two practical findings worth keeping.**

- **`gensim` has no build for Python 3.14**, so Skip-gram is written out from Lab 3 in `src/skipgram.py`, using negative sampling rather than the lab's full softmax (scoring all 17,701 words per pair would take hours). The vectors are sound: `১৯৭১ → মার্চ, জিয়াউর, ১৯৭২`; `সরকার → মুজিবনগর, প্রবাসী`; and they cover 94% of dev words. Each seed trains its own vectors, because the training itself is random.
- **BoW + SVM never converges**, at 5,000 iterations or 20,000, because raw counts here run to 55 with no upper bound. TF-IDF converges easily since it scales everything to 1. Its row is therefore flagged in the output and in the log rather than quietly reported — a concrete demonstration of why the IDF weighting earns its place.

**The lab-versus-library check (§7.1 asks for it):** our hand-written Naive Bayes and scikit-learn's `MultinomialNB` predict **the same label on 100% of 200 dev records**, and score identically (0.487). Run it with `python src/train_classical.py --lab-check`.

**How a record's three parts feed the simple models:** fit a separate vectorizer for the passage, the question, and the answer, then stack all three side by side. That way the model can treat a word differently depending on which part it appeared in. For no-passage records, the passage block is just zeros. What a bag-of-words model *still* can't see is whether the answer's words actually match the passage — that's what M11 and M12 add.

**We use ready-made libraries instead of writing everything by hand, unlike the lab.** The lab writes Naive Bayes, Logistic Regression, and Skip-gram completely from scratch; we use the library versions for speed. As a sanity check, run both versions on 200 sample records and confirm they predict the same labels — write the agreement number in the log. Where no library exists for something the lab covers, we still write it by hand: the edit-distance calculation (Lab 1), the n-gram language model (Lab 2), attention (Lab 4), and positional encoding (Lab 5).

### 7.1a M11 — the character-level language model

This follows the lab's n-gram model exactly (predicting the next item from a fixed history, using counts), but at the **character** level (our answers are only 16 characters long on average — too short for word-level n-grams), with **Laplace smoothing** added so an unseen sequence never gets probability zero.

```
P(next_char | history) = (count(history, next_char) + 1) / (count(history) + alphabet_size)
score(text) = average of log P(each character | its history)
```

**Settings, as measured (`python src/features.py --tune-lm`).** History length and smoothing were chosen by AUC on the **train** hard subset, then confirmed on dev:

| History length | Smoothing k | train hard AUC | dev hard AUC | |
|---|---|---|---|---|
| 3 | 1.0 (plain Laplace) | 0.606 | 0.640 | the lab's own settings |
| 3 | 0.1 | 0.605 | 0.641 | |
| 4 | 1.0 | 0.603 | 0.640 | |
| **4** | **0.1** | **0.611** | **0.657** | **in use** |

Laplace (add 1) turned out too blunt here: with about 60 different characters, a character the passage *has* seen scores only twice one it has never seen, so a copied answer and an invented one look nearly alike. Adding 0.1 instead sharpens that contrast. Add-k is the standard generalisation of the lab's add-1.

| How it's used | What it does | What comes out |
|---|---|---|
| **(a) Score an answer against its own passage** | Build a tiny language model from just *that record's* passage, then score the answer against it | A number: "how much does this answer sound like it came from this passage?" This is a softer version of exact string-matching that survives small spelling changes. Used as a feature in M12 and as a simple rule of its own. |
| **(b) A standalone classifier** | Train one model on all correct train answers, another on all wrong train answers; predict "correct" if the answer scores higher against the correct-answer model | A label — but this model only ever sees the *answer text*, nothing else. Compare it against the "answer-only" sanity check (§9.2). If it scores much higher than that check, it just learned the writing style of whoever wrote the wrong answers, not whether they're actually wrong. |

(a) only ever looks at that one record's own passage, so it doesn't break the no-searching rule. (b) is a normal trained model, like Naive Bayes — not a lookup table.

### 7.1b M12 — similarity features, and the "fuzzy" baseline

Every one of these features compares a record **to itself** — never to any other record.

| Feature | What it measures | Comes from |
|---|---|---|
| `exact_in_passage` | Does the cleaned answer appear word-for-word in the cleaned passage? | Lab 2 |
| `edit_passage` | Smallest normalised edit distance between the answer and any same-length window of the passage (0 = exact match, 1 = nothing alike) | Lab 1 |
| `edit_question` | Same idea, but comparing the answer to the question | Lab 1 |
| `token_overlap` | What share of the answer's words also appear in the passage | Lab 2 |
| `cos_passage`, `cos_question` | Cosine similarity between the answer's average word vector and the passage's / question's | Lab 3 |
| `lm_passage` | The M11(a) score | Lab 2 |
| `has_context` | Just 1 or 0 — whether there's a passage at all | — |
| `answer_len`, `digit_share` | Answer length, and share of characters that are digits | — |

**How `edit_passage` is actually computed (built, and better than the sketch above).** Comparing the answer against every window of a 3,127-character passage would take billions of steps. Instead `best_match_distance()` uses the standard "free start" edit-distance table: the first row is all zeros, so the match may begin anywhere, and one pass finds the closest piece of the passage *of any length*. That is not only faster but more correct — the window version misses a better match that is a little longer or shorter (it returned 5 edits on a real record where the true answer is 4). The distance is divided by the answer's length, which is the most edits it can ever need.

Two checks back this up, both inside `python src/features.py --check`:
- the plain Lab 1 table agrees with `rapidfuzz` on 500 real answer pairs (0 disagreements);
- the fast search agrees with brute-force checking of *every* substring, on random small cases.

**One detail the data forced.** 198 train answers end in a Bengali full stop ("আরবি।") where the passage has them without it. So both `exact_in_passage` and `edit_passage` trim a trailing "।" first; before that fix the two features disagreed on exactly those records.

Feed these features (standardised) into a plain `LogisticRegression`. This model has no randomness, so run it once. Look at which feature got the biggest weight — that tells you what kind of similarity the model actually relies on.

**V6 — the "fuzzy" string-matching baseline.** Predict "correct" if `edit_passage` is below a threshold, chosen on the train split only. **Measured** (`python src/features.py --baseline`) — and the answer turned out to be two thresholds, not one:

| Tuned on | threshold | dev all | dev easy | dev **hard** |
|---|---|---|---|---|
| Exact matcher (no fuzziness) | — | 0.850 | 0.996 | 0.487 |
| All train has-context records | 0.000 | 0.850 | 0.996 | 0.487 |
| Train **hard** records only | 0.405 | 0.572 | 0.543 | **0.591** |

Tuning on everything picks a threshold of 0 — it decides fuzziness doesn't pay. That's right *overall*: easy records are already solved by exact matching, so any allowance just lets wrong answers through. But on hard records, where exact matching has nothing to work with, allowing about 40% of the characters to differ lifts the score from 0.487 to **0.591**.

**So the bar on the hard subset is 0.591, not 0.487.** Report the best rule on each row. A model that beats exact matching on hard but not 0.591 has only learned fuzzy string matching. (These are dev numbers, n = 706 has-context and 218 hard; the corpus-wide figures in §3.3 cover different records.)

**What the features are actually worth (dev, from `--check`).** AUC is the chance a feature ranks a random correct answer above a random wrong one; 0.50 is useless. `edit_passage` is a *distance*, so for it low is good and its AUC reads below 0.50.

| Feature | has-context AUC | **hard-subset AUC** |
|---|---|---|
| `exact_in_passage` | 0.851 | 0.528 — collapses, exactly as intended |
| `token_overlap` | 0.774 | 0.613 |
| `lm_passage` (M11a) | 0.700 | **0.657 — the best single feature on hard** |
| `edit_passage` | 0.183 (0.817 inverted) | 0.400 (0.600 inverted) |
| `answer_len` | 0.526 | 0.563 |

This is the justification for M11 and M12 existing: on the hard subset the exact matcher is dead (0.528) while the graded features still carry real signal. The character language model does best there.

**The M11(b) answer-style classifier scores 0.529–0.546 on dev**, next to the answer-only probe's 0.542 — i.e. no better than reading the answer alone. Good news: the wrong answers carry no strong writing fingerprint that a model could exploit instead of checking the facts.

**A quick sanity check on the word vectors (from Lab 3):** before using the Skip-gram vectors for anything, print the 5 nearest words (by cosine similarity) to about 10 common Bengali words (river, king, year, science). If the neighbours look unrelated, the vectors aren't good enough to trust. This is a one-time offline check — it's never used to fetch anything at prediction time.

### 7.1c M3 — the recurrent models, built in PyTorch

**Setting up the vocabulary and word vectors (Lab 4):**

- Use the §6 tokenizer. Build the vocabulary from the **train split only**, keeping words seen at least twice.
- Reserve ids: `<PAD>` = 0, `<UNK>` = 1, `<SEP>` = 2.
- For each word, use its Skip-gram vector (M2) if it has one; otherwise a random vector; `<PAD>` gets all zeros. Load it with `nn.Embedding.from_pretrained(matrix, freeze=False, padding_idx=0)`.

**How input is formatted:** `passage <SEP> question <SEP> answer`, capped at 256 tokens. If it's too long, cut the **passage** — never the question or answer. If there's no passage: `question <SEP> answer`.

**A mistake the lab's code makes, that we fix here:** the lab reads the RNN's very last hidden state, even though it was padded out with `<PAD>` tokens — so a short sentence's "final state" is actually mostly padding noise. We instead "pack" the batch (`pack_padded_sequence(...)`), so the model's final state is always the state at the sequence's *real* last word.

**The output layer:** take the final hidden state (both directions, for bidirectional models), apply 30% dropout, then a single linear output. Use `BCEWithLogitsLoss`; the sigmoid of the output equals **P(correct)** — matching our `1 = correct` rule.

**Dot-product attention** (following the lab's use of `torch.bmm`):

```python
q = self.query(h_final).unsqueeze(1)                          # (batch, 1, hidden)
scores = torch.bmm(q, outputs.transpose(1, 2)) / H ** 0.5      # (batch, 1, time)
scores = scores.masked_fill(pad_mask.unsqueeze(1), -1e9)       # ignore padding
weights = torch.softmax(scores, dim=-1)
context = torch.bmm(weights, outputs).squeeze(1)
logit = self.fc(torch.cat([context, h_final], dim=1))
```

Save the attention weights for 20 dev examples (10 correct, 10 hallucinated, half of them hard) and check which passage words got the most attention. This is the first model on the ladder that can show *where* it looked.

**The training loop** (written by hand, the way the lab does it — not the Hugging Face `Trainer`):
`optimizer.zero_grad()` → forward pass → compute loss → `loss.backward()` → clip gradients → `optimizer.step()`.

| Setting | Value |
|---|---|
| Optimizer | Adam, learning rate 1e-3 |
| Batch size | 32, shuffled using the run's seed |
| Epochs | Up to 20, stop early if dev loss stalls for 3 epochs, keep the best checkpoint |
| Seeds | 42, 1337, 2024 |

### 7.1d M13 — a Transformer built completely from scratch

This is the lab's own `TransformerClassifier`, applied to the real task. It has the **same kind of design** as BanglaBERT, but **none of BanglaBERT's pretraining** — so comparing the two tells us exactly how much of BanglaBERT's score comes from its architecture versus its pretraining.

| Part | Setting |
|---|---|
| Vocabulary, input, truncation | Same as M3 (§7.1c) — only the architecture is different |
| Word embeddings | `nn.Embedding(vocab_size, 128, padding_idx=0)`, random start, scaled up before adding position info |
| Position information | Fixed sine/cosine pattern (the lab's method) |
| The Transformer itself | `nn.TransformerEncoderLayer(d_model=128, nhead=4, dim_feedforward=256, dropout=0.1, batch_first=True)`, stacked 2 layers deep |
| Padding mask | Tells the model to ignore `<PAD>` positions |
| Combining the output into one vector | **Masked** average — see the fix below |
| Output layer | One number, `BCEWithLogitsLoss`, sigmoid = P(correct) |
| Optimizer | Adam, learning rate 5e-4, with warm-up over the first 10% of training |
| Training | Batch size 32, up to 30 epochs, stop early if dev loss stalls for 3 epochs, 3 seeds |

**A mistake the lab's code makes, that we fix here:** the lab averages *all* positions together, including the padding — fine for its example sentences (all the same length), wrong for our passages (wildly different lengths). Use a masked average instead, so padding contributes nothing:

```python
keep = (~pad_mask).unsqueeze(-1).float()
pooled = (encoded * keep).sum(1) / keep.sum(1).clamp(min=1)
```

**What to expect:** a Transformer with no pretraining, trained on only ~6,000 records, should score somewhere near the classical models — well below the pretrained ones. **That gap is the actual finding.** Don't try to tune it away.

**One more thing about character n-grams:** Bengali words change shape a lot depending on grammar (inflection) and combine freely (compounding), so word-level features can miss related words. Character-level features partly fix this, and also catch near-miss wrong answers that differ by only a letter or two.

**Expected score for classical models: 0.50–0.65.** That's the *correct*, expected result at this data size — these models have no pretraining to lean on. It's not a failure.

### 7.1e What was actually built (M3 + M13)

`src/train_neural.py`, CPU only, PyTorch 2.12.1+cpu. 65 tests in `tests/test_train_neural.py`.

```bash
python src/train_neural.py --check                     # proves the three fixes below
python src/train_neural.py --all --seeds --log         # 5 models x 3 seeds
python src/train_neural.py --model bilstm_attn --attention
```

**Three mistakes in the lab code, each demonstrated by `--check` rather than asserted.**

1. **Padding corrupts the recurrent models' final state.** Lab 4 reads the state at the last
   position of a padded batch, so a short record's "summary" is the state after running over
   `<PAD>`. Every batch here is packed. Measured against running each record alone (no padding
   at all), the packed state matches to 0.000000 while the lab's way is off by ~0.5 per number
   — and agrees only for the single longest record in the batch, the one with no padding.
2. **Lab 5 averages the Transformer's output over padding too.** Harmless for its 8-word toy
   sentences, wrong for inputs of 3–250 tokens. The average here is masked, and `--check`
   confirms a record scores identically alone or in a heavily padded batch.
3. **Not in this guide — found while building.** Lab 5 scales embeddings by `sqrt(d_model)`
   but leaves PyTorch's default `N(0,1)` start, so token vectors come out ~11x larger than the
   position signal (which is ±1) and word order is effectively drowned out. Starting the
   embedding at `std = d_model ** -0.5` puts them level (measured: 1.00 vs 0.67).

**Two settings the guide left open, and why.**

| Setting | Value | Reason |
|---|---|---|
| Gradient clip norm | 5.0 | §7.1c says "clip gradients" without a number; 5.0 is the usual default |
| Hidden size | 128 for every M3 model | the lab's own example size, and keeping it equal across the four means the comparison is about architecture, not capacity |
| CPU threads | 6 | measured on this machine: 1 thread 128 s/epoch, 4: 59, **6: 57**, 10 (torch's default): 64, 16: 161. More threads is slower — the matrices are small, so thread overhead dominates |

**A guard worth knowing about.** A network that finds no signal does not raise an error; it
settles on answering the same thing every time, which scores ~0.333 and reads as a merely bad
model. `collapse_warning()` detects that and says so: `WARNING: predicts correct for 100% of
records - it found no signal`. The run is still logged, with the warning attached.

**Honest caveat on every number in this section.** Dev is used to decide when to stop training
and which epoch to keep, so these dev scores are mildly optimistic. The test split is untouched.

### 7.1f The M3 / M13 results

`python src/train_neural.py --all --seeds --log`, dev split, F2 input, V1 preprocessing,
3 seeds each, CPU. `hard` is the PRD G4 target.

| Model | Lab | dev macro-F1 | has-context + **hard** | epochs | seeds |
|---|---|---|---|---|---|
| Vanilla RNN (`rnn`) | 4 | 0.560 ±0.009 | **0.605** — above the fuzzy rule | 4.7 | 3 |
| BiLSTM + attention (`bilstm_attn`) | 4 | 0.551 ±0.003 | **0.598** — above the fuzzy rule | 5.0 | 3 |
| Stacked BiLSTM (2 layers) (`bilstm`) | 4 | 0.551 ±0.004 | **0.585** | 5.0 | 3 |
| Bidirectional RNN (`birnn`) | 4 | 0.553 ±0.001 | **0.583** | 5.0 | 3 |
| Transformer from scratch (`transformer_scratch`) | 5 | 0.465 ±0.072 ⚠️ found no signal | **0.433** | 5.7 | 3 |
| *the bars to clear* | | | *exact 0.487 · fuzzy 0.591* | | |
| *best classical (tfidf_logreg / tfidf_svm)* | 2–3 | | *0.610* | | |

**Four things this table says.**

1. **The plain RNN is the best of them**, and the only architecture that clears the fuzzy
   rule by a clear margin. Bidirectionality, stacking and attention all cost a little rather
   than gaining: on 6,134 records the extra capacity is spent memorising, not learning.
2. **None of them beats the best classical model** (`tfidf_logreg`, 0.610 on hard). Reading
   the words in order is not, by itself, worth anything here.
3. **The from-scratch Transformer is the weakest thing in the project — and that is the
   point.** It has the same architecture family as BanglaBERT and none of its pretraining.
   Guide 7.1d predicted it would land near the classical models; it lands below them.
4. **It is also the least stable.** Its spread across seeds is ±0.072 where every other
   model sits at ±0.001 to ±0.009, and on seed 2024 it collapsed outright to answering one
   class for nearly everything (0.364). `collapse_warning()` flagged that run rather than
   letting it pass as a merely poor score. An attention model with no pretraining on this
   little data is a coin toss that sometimes lands on its edge.

**What this sets up.** The gap between `transformer_scratch` (0.433 on hard) and a fine-tuned
BanglaBERT is the measurement M4 exists to make: same design, different history.

### 7.2 The pretrained transformer models

| Model | Hugging Face ID | Notes |
|---|---|---|
| **BanglaBERT** | `csebuetnlp/banglabert` | An ELECTRA-style model, Bengali only. Our main Phase 1 model. Needs the special normaliser (§6). |
| BanglishBERT | `csebuetnlp/banglishbert` | Bengali + English. Optional here; becomes central in Phase 2. |
| BanglaBERT large | `csebuetnlp/banglabert_large` | Use if you have enough GPU memory |
| MuRIL | `google/muril-base-cased` | Covers 17 Indian languages. Required by M5. |
| MuRIL large | `google/muril-large-cased` | — |
| XLM-R | `xlm-roberta-base` / `-large` | A strong general-purpose multilingual model |
| mBERT | `bert-base-multilingual-cased` | — |
| IndicBERT v2 | `ai4bharat/IndicBERTv2-MLM-only` | — |

**Why BanglaBERT and MuRIL are our top picks:** BanglaBERT was trained specifically on Bengali; MuRIL is required by M5. XLM-R is a strong general baseline that often wins anyway.

> MuRIL and BanglishBERT are especially strong for *code-mixed* (Bengali+English) text — that's a Phase 2 advantage, not a reason to prefer them here.

**A reality check from a similar published task:** in BanTH's benchmark (transliterated Bangla, a related but different task), seven very different models scored between 74.5 and 77.4 — just a 2.8-point spread. **Don't expect the choice of architecture alone to transform your score.** Input format and further pretraining move the needle more. Report ties honestly, using the McNemar test (§9).

### 7.3 Fine-tuning settings (for the pretrained models, M4–M6 only)

The recurrent models and the from-scratch Transformer use their own settings (§7.1c, §7.1d) — these settings below are only for the *pretrained* encoders.

```python
LEARNING_RATE = 2e-5
OPTIMIZER     = "AdamW"
MAX_LENGTH    = 512      # use 256 instead if your answers are short — much faster
EPOCHS        = 5
BATCH_SIZE    = 32       # drop to 16 if you run out of memory
EARLY_STOPPING = "validation loss"
WARMUP_RATIO  = 0.1
WEIGHT_DECAY  = 0.01
```

A minimal training script:

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

With only ~4,000 examples, a single random split can be misleading about which model is actually better. Use **5-fold cross-validation on train+dev combined**, report the mean and spread, and leave the test set completely untouched until the very end.

Run every setting with all **3 seeds** (42, 1337, 2024) and report the average. A 1-point difference on a single seed is just noise, not a real result.

---

## 8. Further pretraining (the single highest-payoff step)

### 8.1 The recipe (proven on a similar task)

```
Objective:      Masked Language Modelling
Masking rate:   15%
Learning rate:  1e-5
Batch size:     32
Epochs:         5
Corpus:         unlabelled Bengali text
```

### 8.2 What text to further-pretrain on

Use any large collection of unlabelled Bengali text. (BanglaTLit-PT, BanglishRev, and MixSarc are Phase 2 corpora — they're Banglish, not plain Bengali — don't use them here.)

### 8.3 What kind of improvement to expect

On a similar task, further pretraining moved scores like this:

| Model | Before | After | Change |
|---|---|---|---|
| mBERT | 74.97 | **77.36** | **+2.39** |
| BanglishBERT | 75.07 | 77.12 | +2.05 |
| BanglaBERT | 76.50 | 77.12 | +0.62 |
| XLM-R | 77.35 | 77.04 | **−0.31** |

Two lessons: the gain is real but modest (about 2 points, not 20), and it **doesn't always help** — XLM-R actually got slightly worse. **In Phase 1, only further-pretrain mBERT and XLM-R.** BanglaBERT and BanglishBERT use a different kind of pretraining (ELECTRA) and can't be further-pretrained the standard way.

### 8.4 How to do it, in code

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

Then fine-tune this result exactly as in §7.3.

> **Why BanglaBERT can't use this method:** BanglaBERT and BanglishBERT are a different kind of model (ELECTRA discriminators), not the standard kind this masked-language-modelling method expects. Trying to load them with `AutoModelForMaskedLM` will fail. Stick to mBERT and XLM-R for this step.

---

## 9. The safety/shortcut check — run this before trusting any score

### 9.1 The main check: can a model guess the label from surface features alone?

Train a plain logistic regression on features that contain **no actual words** — just shape:

```python
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

def shortcut_features(text):
    toks = text.split()
    n = max(len(toks), 1)
    return [
        len(toks),                                            # word count
        len(text),                                            # character count
        np.mean([len(t) for t in toks]),                      # average word length
        sum(c.isascii() and c.isalpha() for c in text) / max(len(text),1),  # share of English letters
        text.count(','), text.count('.'), text.count('?'),
        text.count('!'), text.count('"'),
        sum(c.isdigit() for c in text) / max(len(text),1),    # share of digits
        sum(1 for t in toks if t.isupper()) / n,              # share of all-caps words
    ]

X = np.array([shortcut_features(t) for t in df.candidate_answer])
clf = LogisticRegression(max_iter=1000).fit(X_train, y_train)
score = f1_score(y_test, clf.predict(X_test), average="macro")
print(f"SHORTCUT PROBE: {score:.3f}")
```

**How to read the score:**

| Score | What it means |
|---|---|
| Below 0.55 | Clean. Go ahead. |
| 0.55 – 0.60 | Borderline. Look at `clf.coef_` to find which feature is leaking signal. |
| Above 0.60 | Something is broken. Fix the data and rebuild — don't just lower this bar. |

### 9.2 Three more checks worth doing

**Check 1 — the "answer only" test.** Train a full model using *only* the answer text, no question, no passage. On has-context data, this should score clearly worse than a model given everything. If it doesn't, the answer text alone is giving away the label, and the model isn't checking grounding at all. (M11's second use, §7.1a, is exactly this kind of "answer only" model — report it alongside this check.)

**Check 2 — a human-written holdout.** If you have any test items whose wrong answer was written by a person rather than by AI, keep them aside. If the model does much worse on those, it may have learned the AI's specific writing "fingerprint" rather than genuine hallucination signals.

**Check 3 — suspiciously high scores.** Any macro-F1 above 0.95 on has-context data should be treated as a probable data leak until proven otherwise.

---

## 10. Trying to improve the score, in order of how much it usually helps

Work down this list. Stop once you hit your target.

1. **Sweep input formats and text normalisation.** Try F1/F2/F3, and normalised-vs-raw text, on your best model. Often worth 3–6 points. About half a day of work.
2. **Further pretraining** (§8). Worth roughly 2 points, on the right base model.
3. **Add more hard examples.** If a model scores 0.90 on easy items but only 0.55 on hard ones, adding more hard examples to training usually helps more than changing the architecture.
4. **Combine your top 3 models.** Average their predicted probabilities. Usually worth 1–3 points.
   ```python
   probs = (p_banglabert + p_muril + p_xlmr) / 3
   preds = (probs[:, 1] > threshold).astype(int)   # column 1 = P(correct)
   ```
5. **Tune the decision threshold — on dev only, never 0.5 by default.**
   ```python
   best_t = max(np.arange(0.2, 0.81, 0.01),
                key=lambda t: f1_score(y_dev, (p_dev[:,1] > t).astype(int), average="macro"))
   ```
   Apply that threshold to test exactly once. Usually worth 1–2 points.
6. Adjust class weights, but only if your data drifts away from 50/50.
7. **Hyperparameter search** — the lowest payoff of everything on this list. Don't spend more than a day here.

---

## 11. Using an AI model as a reference point (M8)

This isn't a competitor to beat — it's a **ceiling marker**, and useful evidence for reviewers. Budget about one afternoon.

Try a strong general AI model (GPT-4-class, Gemini, or Llama), zero-shot and few-shot, on 300–500 items. Develop and compare your prompts on **dev**; only run it on **test** as part of the single final M6 evaluation.

**No searching in the prompt.** The prompt should contain only that record's own question, passage (if any), and answer, plus fixed instructions. If you use few-shot examples, use the **same fixed set** of examples for every single item (e.g., pick 4 train records once, using seed 42, and reuse them every time). Never pick different examples per item based on similarity — that's searching. Never give the model a browsing or search tool.

**Prompting styles worth testing:**

| Style | What you add to the prompt |
|---|---|
| Plain | Nothing extra |
| Step-by-step | "Let's think step by step" |
| Explain first | "Explain why, then answer" |

(A "translate first" strategy that works well on transliterated text doesn't apply here — our input is already plain Bengali script, so there's nothing to translate.)

**What to expect:** fine-tuned models usually beat zero-shot prompting by a wide margin — roughly 8 points on a similar task. If a zero-shot AI model somehow beats your own fine-tuned model by a lot, that's a signal your training data might be too small or too noisy.

---

## 12. Scoring and reporting results

### 12.1 What to measure

Main metric: **macro-F1** (treats both classes equally). Also report accuracy, precision/recall per class, and AUC.

Report all of these **separately**, never blended together:
- With passage vs. without passage
- Easy vs. hard
- By subject
- By error type — **dev and test only**, and say so; skip anything still marked `unlabeled`
- Human-written vs. AI-written wrong answers (if you have any human-written ones)

Every has-context row must show **both** baselines: the exact string-matching rule, and the fuzzy one (V6).

**Built** (`src/evaluate.py`). Every model reports through `report(records, predictions, scores)`, so no model can be scored on its own terms. Three choices in it are worth knowing:

- **The 95% interval resamples pairs, not records.** The two answers to one question share a passage and a question, so they rise and fall together. Drawing single records would pretend they are independent and make the interval look tighter than it is.
- **Per-type slices report a detection rate, not macro-F1.** A type slice ("all the `numeric` errors") holds only wrong answers, so macro-F1 on it is degenerate. The table gives the share of that type the model caught, plus a *pair* score that adds the matching correct answers back so it can be compared with the other macro-F1 numbers.
- **McNemar is written out by hand**, because `statsmodels` is not one of this project's dependencies.

There is also an explicit **`has-context + hard`** row, because that is the PRD G4 target and it is *not* the same as the plain "hard" row (which also counts hard no-context records: 0.487 vs 0.440 for the exact rule on dev).

**The human-written vs generated split cannot be reported.** Every record in this corpus is `llm_generated` (`data/SOURCES.md`), so there is no human-written slice to compare against. `report()` prints that instead of leaving a silent gap.

**Table 5 measured on dev** (`python src/evaluate.py --baselines --table`):

| Rule | overall | has-ctx | no-ctx | easy | hard |
|---|---|---|---|---|---|
| Exact string match | 0.678 | 0.850 | 0.333 | 0.767 | 0.440 |
| Fuzzy string match (V6) | 0.558 | 0.572 | 0.333 | 0.562 | 0.539 |
| Always says correct | 0.333 | 0.333 | 0.333 | 0.333 | 0.333 |
| Always says hallucinated | 0.333 | 0.333 | 0.333 | 0.333 | 0.333 |

The rules score 0.333 on every no-context record, because without a passage they have nothing to check — which is precisely why no-context is the harder half.

### 12.1a M14 — does word order actually matter to each model?

Lab 3 shows that a bag-of-words model gives "the dog bit the man" and "the man bit the dog" the exact same vector. This test measures what that limitation actually costs us. **Dev only.**

1. Take a trained model's normal predictions on dev.
2. Shuffle the word order inside the passage, question, and answer (each shuffled separately, using seed 42), then predict again with the *same* model.
3. Report the change in score (shuffled score − normal score), broken down overall, has-context, hard, and by error type.

| What we expect | What it means |
|---|---|
| Exactly zero change for the Bag-of-Words / TF-IDF models and the averaged-embedding models | Expected, by design — **if this isn't exactly zero, there's a bug somewhere** |
| A drop for the RNN/LSTM models, the Transformer, and the pretrained models | Good — it means the model is actually using word order |
| The biggest drop should be on `relational` and `contradiction` errors | Those errors *are* about word order and negation |

If a sequence model shows almost no change, that's worth reporting — it means the model isn't really using the word order it was designed to use.

### 12.2 Being statistically careful

- Always run 3 seeds, report the average and spread.
- Use a **McNemar test** to compare two models on the same test set.
- Report a **bootstrap 95% confidence interval** on your headline number (1,000 resamples).

```python
from statsmodels.stats.contingency_tables import mcnemar
# table = [[both_correct, only_A_correct], [only_B_correct, both_wrong]]
print(mcnemar(table, exact=False, correction=True))
```

Without this, you can't honestly claim model A beats model B if they only differ by 1.5 points.

### 12.3 What scores to realistically expect

| Setting | Realistic range | When to double-check for a leak |
|---|---|---|
| Has-context | 0.80 – 0.90 | Above 0.95 |
| No-context | 0.60 – 0.75 | Above 0.85 |
| Hard items only | 0.55 – 0.70 | — |
| Classical models (M1, M2, M11) | 0.50 – 0.65 | Expected, not a problem |
| Similarity features + Logistic Regression (M12), has-context | Around the fuzzy baseline (V6) | Far above V6 → check for a leak |
| Recurrent models (M3) | 0.55 – 0.70 | — |
| From-scratch Transformer (M13) | Near M3, well below the pretrained models | Above the pretrained models → check for a leak |
| Zero-shot AI model | 0.60 – 0.72 | — |

**No-context scores will be noticeably lower than has-context scores. That's expected, not a bug** — answering from memory with no source text is a genuinely harder task.

### 12.4 The result tables to produce

**Table 1 — main results** (in the order models were built)

| Model | Lab | Has-context | Has-context (hard only) | No-context | Overall | Accuracy |
|---|---|---|---|---|---|---|
| *Reference rules* | | | | | | |
| Always guess the more common label | — | | | | | |
| Exact string match | 2 | | | — | | |
| Fuzzy string match (V6) | 1 | | | — | | |
| Passage language-model rule (M11a) | 2 | | | — | | |
| *Simple word-counting (M1)* | | | | | | |
| Bag of Words + Naive Bayes | 2, 3 | | | | | |
| TF-IDF + Naive Bayes | 2, 3 | | | | | |
| TF-IDF + Logistic Regression | 2, 3 | | | | | |
| TF-IDF + SVM | 2 | | | | | |
| *Language model (M11)* | | | | | | |
| Answer-only classifier | 2, 3 | | | | | |
| *Word embeddings (M2)* | | | | | | |
| Skip-gram average + Logistic Regression | 3 | | | | | |
| Skip-gram TF-IDF-weighted + Logistic Regression | 3 | | | | | |
| Skip-gram + XGBoost (best version) | 3 | | | | | |
| *Similarity features (M12)* | | | | | | |
| Edit distance + cosine + LM + Logistic Regression | 1, 2, 3 | | | | | |
| *Recurrent models (M3)* | | | | | | |
| Plain RNN | 4 | | | | | |
| Bidirectional RNN | 4 | | | | | |
| BiLSTM | 4 | | | | | |
| BiLSTM + attention | 4 | | | | | |
| *Transformer from scratch (M13)* | 5 | | | | | |
| *Pretrained models (M4/M5)* | | | | | | |
| mBERT / XLM-R / MuRIL / BanglaBERT / IndicBERT v2 | — | | | | | |
| *Further pretrained (M6)* | | | | | | |
| FPT-mBERT / FPT-XLM-R (ours) | — | | | | | |
| *Combined models (M7)* | | | | | | |
| Top-3 average | — | | | | | |
| *AI reference (M8)* | | | | | | |
| Zero-shot AI model | — | | | | | |

**Table 2** — easy vs. hard, per model
**Table 3** — per error-type score (dev/test only)
**Table 4** — normalisation × input-format experiment
**Table 5** — safety-check results (surface-feature check, answer-only check, both string baselines)
**Table 6** — cleanup experiment (M10): the V0–V4 + V2-demo variants ✅ **done** → `results/tables/table6_preprocessing_ablation.csv`, written up in §6
**Table 7** — word-order test (M14): score change per model, per error type
**Table 8** — per-subject score, for the best model at each rung of the ladder

---

## 13. A rough timeline

*(Weeks 1–3 below describe how the dataset itself was built — that part is already finished. Weeks 4–6 are what's left to do.)*

| Week | What happened / what's next |
|---|---|
| 1–3 *(done)* | Licences checked, schema frozen, corpus built and filtered, annotation done, human agreement measured, splits locked |
| **4** *(done)* | ✅ `text_bn.py`, `features.py` (M11/M12 + V6 baseline), `preprocess.py`, `evaluate.py`, and `train_classical.py` + `skipgram.py` — M1, M2, M11 and M12 are all trained (Table 5), and the M10 cleanup experiment is run (Table 6) |
| **5** | Build M3 (recurrent models) and M13 (Transformer from scratch); start the first pretrained models (M4/M5); sweep input formats (M9) |
| **6** | Further pretraining; combining models; threshold tuning; AI reference point; the word-order test (M14); final safety check; produce all 8 result tables |

---

## 14. Risk register

| Risk | How likely | How bad | What to do about it |
|---|---|---|---|
| A shortcut inflates scores without real learning | High | Critical | Run the safety check (§9) before trusting anything |
| Low human agreement | Medium | High | Fix the guidelines, then re-test on the same 100 items |
| The Kaggle dataset turns out unusable | Medium | Low | Don't build on it — use it only as an outside comparison point |
| No-context scores end up low | High | Low | Expected — report it honestly, it's a real finding |
| Models score too close together to tell apart | Medium | Medium | Use 3 seeds + McNemar; report confidence intervals |
| Someone else publishes similar work first | Medium | Medium | This is a course deliverable, not a race |
| A checker drops out | Medium | Medium | We already have both checkers finished |
| Free GPU quota runs out | Low | Medium | Use base-size models, shorter inputs, mixed precision |

---

## 15. The demo interface

Built after the classical ladder, so that the models can be used and not only tabulated.
Beginner-level walkthrough in §18.9 of `PROJECT_WALKTHROUGH.md`.

```bash
python src/serving.py --build     # once, ~6 min CPU: fit all 12 models, cache them
python src/app.py                 # http://127.0.0.1:8000
```

Flask, Jinja2 and joblib only — no metrics stack, no containers, nothing to start besides the
one server.

### 15.1 What it serves

| Page | Shows |
|---|---|
| `/` | Type question + answer (+ optional passage). All 12 models judge it, ranked by dev hard score, beside the M12 evidence and both rule baselines |
| `/models` | The scoreboard, read from `results/experiment_log.csv` and averaged over seeds |
| `/health` | Readiness check (models loaded or not) |

Two-column layout: input on the left, results on the right. Measured at **576px** of height,
so it fits a laptop screen without scrolling.

### 15.2 Three rules the implementation follows

1. **Serving and training are one code path.** `train_and_predict` was split into
   `fit_model()` and `FittedModel.predict()`; the demo calls `fit_model()`. If serving rebuilt
   the feature pipeline itself, the two would drift the first time either changed and the demo
   would show predictions from a model nobody measured. The split was verified
   behaviour-preserving: `tfidf_logreg` still scores 0.549 / 0.610 with 215,601 features.
2. **The page owns no numbers.** Every score comes from the experiment log. A model run on
   three seeds is shown as the mean, so the page cannot contradict Table 5.
3. **The test split is unreachable.** A test parses `serving.py` and `app.py` with the AST —
   stripping docstrings, which discuss the rule at length — and fails if either loads `test`.

### 15.3 What the demo is good for pedagogically

The clearest single screen is a `numeric` hallucination: a passage saying ৬৫টি against an
answer saying ৬৭টি. **Eight of the twelve models call it correct**, and so does the fuzzy
matcher — only 2% of the answer's characters differ. Only the exact-match rule gets it right,
and only because the string "৬৭টি" is absent.

It demonstrates the project's premise in one view: these models compare shapes of text, and
none of them can read a number out of a passage and check it against a number in an answer.

---

## 16. Background reading

**Read these first:**
- **BanTH** (arXiv 2410.13281) — transliterated Bangla, further-pretraining recipe, full baseline table. Our closest template.
- **BenHalluEval** (arXiv 2605.31483) — the Bengali hallucination benchmark our results will be compared against.
- **The "অলীকবচন" Kaggle competition** — possible source of some of our raw data; licence still unresolved (PRD Q1).

**Models:**
- BanglaBERT / BanglishBERT — `github.com/csebuetnlp/banglabert`
- MuRIL — `google/muril-base-cased`

**Related hallucination-detection work:**
- MedHallu, SHROOM-CAP, HaluEval, HalluLens — background on how hallucinations are typically categorised and measured elsewhere.

---

## Appendix A — Where each lab topic ended up in this project

The full list of what's used and what's skipped, with reasons, is in PRD §5.3b. This table just says *where in the code* each used topic lives.

| Lab | Lab's version | This project's version | What's different |
|---|---|---|---|
| 1 | Cleaning with regex | `text_bn.py` → `clean()` (§6) | Bengali-safe patterns; keeps digits and Bengali letters; adds Unicode normalisation |
| 1 | `word_tokenize` | `text_bn.py` → `tokenize()` (§6) | A regex tokenizer for Bengali; the Bengali full stop is its own token |
| 1 | English stop words | `configs/bn_stopwords.txt` (M10 only) | A published Bengali list; negation and number words protected via `bn_protected_words.txt` |
| 1 | Porter Stemmer | `text_bn.py` → `stem_word()` (M10 only) | Bengali noun endings; the stem must be a real train word; never touches negation |
| 1 | Edit-distance calculation | `features.py` (M12, V6) | Normalised; "free start" table finds the closest piece of the passage |
| 2 | Bag of Words, TF-IDF | `train_classical.py` (M1) | One vectorizer per record part (passage/question/answer) |
| 2 | N-gram language model | `features.py` (M11) | Character-level; add-k smoothing (k tuned on train); used to *score*, not generate |
| 3 | Skip-gram from scratch | gensim's Word2Vec (M2) | Trained on the train split; sanity-checked with nearest neighbours |
| 3 | Cosine similarity | `features.py` (M12) | Only compares an answer to its own passage/question — added once Skip-gram (M2) exists |
| 3 | Naive Bayes, Logistic Regression (hand-written) | Library versions (M1, M2, M12) | Checked against the hand-written version on a sample |
| 3 | Averaging / TF-IDF-weighting embeddings | (M2) | Same idea, unchanged |
| 3 | "Dog bit the man" word-order test | `evaluate.py` (M14) | Run on every trained model, reported per error type |
| 4 | Loading pretrained vectors, PAD/UNK | `train_neural.py` (M3) | Uses our own Bengali Skip-gram vectors |
| 4 | RNN / BiLSTM classifiers | `train_neural.py` (M3) | Uses packed sequences, so padding doesn't blur the result |
| 4 | Attention (`torch.bmm`) | `train_neural.py` (M3) | Padding is masked out; weights are saved for inspection |
| 4 | `BCEWithLogitsLoss` | `train_neural.py` (M3, M13) | Sigmoid output = P(correct) |
| 5 | Positional encoding, Transformer classifier | `train_neural.py` (M13) | Uses masked average pooling; passage is truncated first |

## Appendix B — Day-one checklist (status as of 17 September 2026)

- [x] Project folder set up (see §1.3)
- [x] `results/experiment_log.csv` created, with headers
- [x] `data/SOURCES.md` written — ⚠️ two licence questions still open (PRD Q1–Q3)
- [ ] Kaggle competition licence confirmed — still open (PRD Q1)
- [x] Record schema frozen, written to `configs/schema.json`
- [x] Correct/wrong label rule confirmed everywhere (§2.1)
- [—] "Prompt used to generate data" — doesn't apply; Phase 1 generates no data. (The prompt that built the *original* source pool is unknown — PRD Q4)
- [x] Labelling rules written before labelling began
- [x] Both human checkers recruited, briefed, and finished (agreement 0.717 on the pilot, 0.865 on the full test set)
- [x] Seeds fixed: 42, 1337, 2024
- [x] Test set locked — will only be opened once, at M6
