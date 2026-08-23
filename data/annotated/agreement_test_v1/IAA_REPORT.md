# Inter-annotator agreement report — M2 gate (DL3)

**Date:** 2026-08-24 · **Corpus:** `data/corpus/bn_v1/` (4,480 pairs) · **Items:** 100, blind
**Annotators:** Tawhidul Hasan (A) · Shejan (B) — both native Bangla speakers, labelling independently

## Result

> ## GATE M2 PASSED — Cohen's κ = 0.717 (required ≥ 0.60)

| Measure | Value |
|---|---|
| Items scored | 92 of 100 |
| Raw agreement | 79 / 92 = **85.9%** |
| **Cohen's κ** | **0.717** |
| Annotator A vs answer key | 89.1% |
| Annotator B vs answer key | 94.6% |

κ measures agreement *above chance*. On a 50/50 task two people guessing already agree half the
time, so raw agreement flatters; κ = 0.717 is "substantial" on the Landis–Koch scale and clears
the gate with room to spare.

```bash
python src/score_agreement.py \
  --a data/annotated/agreement_test_v1/items_for_annotation_tawhid.csv \
  --b data/annotated/agreement_test_v1/items_for_annotation_shejan.csv
```

## Agreement by slice

| Slice | Agreement |
|---|---|
| has-context | 46 / 55 = 84% |
| no-context | 33 / 37 = 89% |
| easy | 61 / 69 = 88% |
| hard | 18 / 23 = 78% |

Hard items agree less than easy ones — the expected direction, since those are the near-miss
pairs where the string shortcut gives no help.

## The 8 unscored items — all flagged `unsure`

**`unsure` here means "this item cannot be labelled honestly", not "the annotator was hesitant."**
Excluding them from κ is correct: an item nobody can label is not a disagreement.

Every one of the eight turned out to be a **defect in the item**, and they cluster into three
kinds:

| Kind | Items | What is wrong |
|---|---|---|
| **The passage does not answer the question** | `item_076`, `item_079`, `item_099` | all `law`. e.g. `item_076` asks when IPC §377 was *introduced* (1861); the passage only covers the 2018 ruling |
| **The answer type does not match the question** | `item_051`, `item_091` | asked **কবে** (*when*); the answer is a whole biographical sentence — "asked when, answered what kind of person" |
| **Broken or unanswerable text** | `item_048`, `item_028`, `item_063` | `item_048`'s question is truncated (`...কিসের শ. ওপর নিভর্র করে? ড়ড় পব`) |

> **Correction to an earlier draft of this report.** It attributed the `unsure` labels — three of
> them `law` — to the cost of keeping lookup-heavy subjects. That was wrong. The annotators were
> not failing to look facts up; they were flagging items whose passage does not answer the
> question. The law clustering is a *passage-matching* problem, not a difficulty problem.

**Measured extent across the whole corpus:** 47 of 2,688 has-context correct answers (**1.7%**)
share less than a fifth of their vocabulary with their own passage. Concentrated in `grammar`
(20), `law` (14), `history` (8). The `grammar` cases are mostly false alarms — the answer is a
category name derivable from the passage rather than quoted from it. So the genuine rate is
roughly **1%**, and the sample of 100 simply caught several at once.

## `item_005` — resolved

An earlier run excluded `item_005` because both sheets held a different candidate answer than the
answer key expected. The annotators corrected it, and the remaining mismatch was cosmetic: the
sheets carried the Bengali numeral `৩` where the corpus stores ASCII `3`. The scorer now folds
Bengali and ASCII digits together before comparing, so the item is scored normally.

Both annotators labelled it `wrong` / `numeric` / `easy`, matching the key. Including it raised κ
from 0.714 to **0.717** and the scored count from 91 to 92.

The integrity check itself stays: it re-verifies every sheet row against the corpus record the key
points at, so a sheet copied from an older build can never silently corrupt a score.

## Data-entry corrections applied

Two unambiguous slips, repaired before scoring. Originals backed up.

| Sheet | Item | Problem | Correction |
|---|---|---|---|
| shejan | `item_003` | the label `correct` was typed into the **answer** cell, leaving `your_label` empty | `your_label = correct`; answer restored to `৩২ বর্গ একক।` |
| tawhid | `item_018` | `your_type` and `your_difficulty` were swapped (`type=easy`, `difficulty=none`) | `type = none`, `difficulty = easy` |

Both sheets are now **100/100 complete** on label, type, and difficulty.

A third artifact was harmless: Excel stripped the leading apostrophe from 8 questions in B's sheet
(`'জোয়ার'` → `জোয়ার'`). The scorer normalises it away.

## What the 13 disagreements taught us

Each cause is now a rule in
[`docs/ANNOTATION_GUIDELINES.md`](../../../docs/ANNOTATION_GUIDELINES.md) (Rules 11–15).

### 1. The passage does not support the answer (5 items)

The single largest cause. A applied Rule 5 strictly ("not in the passage → wrong"); B fell back on
general knowledge.

> **`item_009`** — the passage describes Jagadish Chandra Bose's schooling and says he became
> interested in **physics**. The question asks which *literature* interested him; the answer is
> `বাংলার লোক অভিনয়, যাত্রা-পালাগান, রামায়ণ ও মহাভারত` — true of his life, absent from the passage.
>
> **A was right by our rules.** The source label (`correct`) reflects world knowledge rather than
> the passage. Rule 5 stands; this is a label error in the source data, not an annotator error.

### 2. The two answers differ only by a typo (2 items)

> **`item_014`** — `…সম্প্রীতির শিক্ষা` vs `…সম্প্রীতির শিক্সা`
> **`item_070`** — `…তৃতীয় ওডিআইয়ে` vs `…তৃতীয় ওডিআঈয়ে`

One character apart, no change in meaning — corrupted duplicates, not hallucinations. They collide
with Rule 8 ("do not mark an answer wrong for spelling"), so an annotator following the guidelines
*cannot* label them. Rule 12 now routes them to `unsure`.

**Extent:** 165 pairs (3.7%) differ by 1–2 letters in a subject where spelling is not the answer;
inspection suggests about half are genuine meaning changes (`ক্যাথোডে` vs `অ্যানোডে` is a real
hallucination), so the broken class is around **2%**.

For contrast, 979 pairs (**21.9%**) differ by 1–2 **digits** (`১৬` vs `১৩`, `অনুচ্ছেদ ৩২` vs
`অনুচ্ছেদ ২৭`). Those are legitimate numeric hallucinations and must be kept.

### 3. Literal versus idiomatic meaning (1 item)

> **`item_032`** — *"কপালপোড়া" এর শাব্দিক অর্থ কী?* → `পোড়া কপাল (burnt forehead)`

শাব্দিক অর্থ asks for the **literal** sense, so the gloss is correct even though the idiom means
"unlucky". Now Rule 13.

## Two corpus-quality issues, measured and left unchanged

**1. Wikipedia citation markers — 1,266 records (14.1%), 633 pairs.** `[1][2][3]` survives in the
passage text, almost all in `context` (1,264), concentrated in `reading_comprehension` (1,110) and
`history` (154). Pure noise; stripping it changes no meaning. Rule 15 tells annotators to read
past it.

**2. Answers that are a whole sentence copied from the passage — 217 records (4.0%).**

> **`item_051`** — Q: *রেগি টোরিয়ান কবে জন্মগ্রহণ করেন?* ("when was he born?")
> A: `রেগি টোরিয়ান (জন্ম ২২ এপ্রিল ১৯৭৫) একজন মার্কিন হার্ডলার।`

The birth date *is* inside it, so it is arguably correct — but it reads as a passage dump, which is
why both annotators flagged it. Rule 14 covers it.

**Neither has been applied to the corpus.** Both would change data that a gate has already passed
on, so they are recorded here as decisions for the project owner.

## Verdict

M2 is met. Two independent native speakers using these guidelines reach κ = 0.717, every
disagreement traces to a specific and fixable cause, and every `unsure` traces to a real defect in
the item rather than to annotator uncertainty. Annotation can scale to the full corpus (M3) once
Rules 11–15 and §2C have been read.
