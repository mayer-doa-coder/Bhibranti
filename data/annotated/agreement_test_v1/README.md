# Agreement test v1 — the 100-item check for M2

**What this is for:** testing whether your rulebook (`docs/ANNOTATION_GUIDELINES.md`) actually
works, by having two people use it independently and checking how often they agree.

Built by [`src/build_agreement_test.py`](../../../src/build_agreement_test.py), seed 42,
reproducible. 100 items pulled from `data/generated/pilot_v1/`, one item per source pair (see
"Why one per pair" below).

```bash
python src/build_agreement_test.py
```

## The 3 files here, and who touches which one

| File | Who opens it | When |
|---|---|---|
| `items_for_annotation_BLANK.csv` | **You and your friend** | Now — make 2 copies, one each |
| `answer_key_DO_NOT_OPEN_YET.csv` | **Nobody** (I use it to score) | Only after both of you finish |
| `selection_log.csv` | Just a record of what was picked | Reference only |

**The single most important rule: do not open `answer_key_DO_NOT_OPEN_YET.csv` until both of you
have finished and submitted your labels.** If you look at it first, even by accident, the test
becomes meaningless — it stops measuring whether the guidelines work and starts measuring
whether you remember the answer key.

## Exactly what to do

1. **Make two copies** of `items_for_annotation_BLANK.csv` — e.g. `items_tawhidul.csv` and
   `items_friend.csv`. One person, one file.
2. **Read `docs/ANNOTATION_GUIDELINES.md` first**, both of you, before starting.
3. **Fill in your copy alone.** For every one of the 100 rows:
   - `your_label` → write `correct` or `wrong`
   - `your_type` → if you wrote `wrong`, pick one of the 6 types from the guidelines
     (`entity`, `numeric`, `relational`, `contradiction`, `fabricated`, `overclaim`); if
     `correct`, write `none`
   - `your_difficulty` → `easy` or `hard`
   - `your_notes` → anything that confused you, or anything about the Banglish text itself that
     was hard to read (this draft hasn't been through Task 3 review yet — flag rough spots)
4. **Don't discuss items with each other while working.** No "hey what did you put for #47."
5. **Send me both filled-in copies** when you're both done.
6. I'll compare your answers to the hidden answer key and to each other, and give you one
   number (Cohen's κ).

## Why each item comes from a different pair

Your dataset is built as pairs — one question, one correct answer, one wrong answer. If both
answers to the same question showed up in this 100-item test, you could figure out the label by
comparing them ("this one looks more different, must be the wrong one") without really applying
the rulebook. So this test uses **at most one answer per question** — you're judging each item
completely on its own, the same way it'll work in the real, full-size annotation later.

## What happens with the result

| κ score | Meaning | What happens next |
|---|---|---|
| ≥ 0.60 | Guidelines work | **M2 done.** Move to converting and labeling the rest of the corpus. |
| < 0.60 | Guidelines have gaps | We look at exactly where you two disagreed, add a rule for it, and run a fresh 100-item round |

Below 0.60 is not a bad result — it's the guidelines' job to close that gap, not yours. Nobody
expects the first draft of a rulebook to be perfect.

## Composition (matches your corpus's own targets)

| | correct | wrong | total |
|---|---:|---:|---:|
| has-context | 30 | 30 | 60 |
| no-context | 20 | 20 | 40 |
| **total** | **50** | **50** | **100** |

60/40 has-context/no-context and 50/50 correct/wrong — the same ratios your full corpus targets
(PRD D2, D3), just at this small scale.
