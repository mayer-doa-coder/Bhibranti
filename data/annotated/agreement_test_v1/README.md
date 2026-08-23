# Agreement test v1 — the 100-item check for M2

**Purpose:** find out whether the rulebook (`docs/ANNOTATION_GUIDELINES.md`) actually works, by
having two people use it independently and measuring how often they agree.

All items are **Bangla script**. Built by [`src/build_agreement_test.py`](../../../src/build_agreement_test.py),
seed 42, reproducible.

## Files, and who opens which

| File | Who opens it | When |
|---|---|---|
| `items_for_annotation_BLANK.csv` | **You and your co-annotator** | Now — make 2 copies, one each |
| `answer_key_DO_NOT_OPEN_YET.csv` | **Nobody**, until both of you finish | Used to score you |

**Do not open the answer key until both sheets are finished.** If you see it first, the test
stops measuring the guidelines and starts measuring your memory.

## Steps

1. Copy `items_for_annotation_BLANK.csv` twice — e.g. `items_tawhidul.csv`, `items_friend.csv`.
   (Files named `items_*.csv` are git-ignored, so two people can work in one clone.)
2. Both read `docs/ANNOTATION_GUIDELINES.md` first.
3. Fill in your own copy, alone. For each of the 100 rows:
   - `your_label` → `correct` or `wrong`; `unsure` if you cannot establish it even after
     checking; `unreadable` if the text is broken
   - `your_type` → `none` if correct; otherwise `entity`, `numeric`, `relational`,
     `contradiction`, `fabricated`, or `overclaim`
   - `your_difficulty` → `easy` or `hard`
   - `your_notes` → anything confusing, and note it whenever you looked something up
4. Do not discuss items with each other while working.
5. Score it:

```bash
python src/score_agreement.py --a data/annotated/agreement_test_v1/items_tawhidul.csv \
                              --b data/annotated/agreement_test_v1/items_friend.csv
```

## What the result means

| Cohen's κ | Meaning | Next step |
|---|---|---|
| ≥ 0.60 | The guidelines work | **M2 passed.** Start annotating the full corpus |
| < 0.60 | The guidelines have gaps | Add a rule for each disagreement, rerun on the same 100 items |

κ below 0.60 is not a bad result and not the annotators' fault. It is the guidelines' job to
close that gap. The scorer prints every disagreement so you know exactly what to write a rule for.

## Why one item per question

Each pair in the corpus is one question with a correct answer and a hallucinated answer. If both
appeared here you could work out the labels by comparing them, instead of applying the rulebook.
So this test uses **at most one answer per question** — the same way real annotation works.

## Composition

Matches the corpus it is drawn from: 60/40 has-context, 50/50 correct/wrong, the corpus's own
share of hard items, and all 13 subjects — so the κ you get describes the data you will actually
annotate, including the harder general-knowledge subjects.

| | correct | wrong | |
|---|---:|---:|---|
| has-context | 30 (10 hard) | 30 (10 hard) | 60 |
| no-context | 20 (4 hard) | 20 (4 hard) | 40 |
| | **50** | **50** | **100** |
