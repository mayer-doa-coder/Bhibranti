# Round 1 — the real annotation (M3)

Built by [`src/build_annotation_sheets.py`](../../../src/build_annotation_sheets.py), seed 42,
reproducible. **Read [`docs/ANNOTATION_GUIDELINES.md`](../../../docs/ANNOTATION_GUIDELINES.md)
first** — §2C and Rules 11–15 are new since the agreement test.

## Two different jobs

| Folder | Rows | Who | Arrives |
|---|---:|---|---|
| `test/tawhid/` | 1,356 | **Tawhid** | **Empty. You label it blind.** |
| `test/shejan/` | 1,356 | **Shejan** | **Empty. You label it blind.** |
| `dev/` | 1,346 | either one | **90% pre-filled.** You verify. |
| `train_spotcheck/` | 1,252 | either one | **90% pre-filled.** You verify. |

### Why test is not pre-filled

`your_type` **is** the binary label: it reads `none` for every correct answer and a real type for
every wrong one. Pre-filling it would tell you the answer on all 1,356 rows and destroy the blind
double-annotation PRD D4 requires. So test costs what it costs.

### What "pre-filled" means for dev and train

- `your_label` — copied from the existing data. **Not a new judgement.** Your job is to catch the
  ones that are wrong.
- `your_type` — derived only where the rule is objective (the numbers differ; a no-context answer
  is simply a different answer). Where it needs a human it is left **blank** — 219 in dev,
  228 in train.
- `your_difficulty` — derived from how close the two answers are.
- **10% of rows are left completely empty on purpose.** They are the attention check. If your
  corrections on those look like your corrections everywhere else, it shows you were reading.

## Filling a row

| Column | What to put |
|---|---|
| `your_label` | `correct` · `wrong` · `unsure` · `unreadable` |
| `your_type` | `none` if correct. If wrong **and there is a passage**: `entity` · `numeric` · `relational` · `contradiction`. If wrong **and there is no passage**: `fabricated` · `overclaim` |
| `your_difficulty` | `easy` · `hard` |
| `your_notes` | a short reason whenever you write `unsure` |

Do not edit `item_id`, `condition`, `context`, `question`, `answer`. The checker compares them
against the corpus and flags any change.

## Check your work

```bash
python src/validate_annotation.py --dir data/annotated/round1/dev
```

Shows progress and catches the mistakes that actually happened last round — a label typed into
the answer cell, `type` and `difficulty` swapped, a wrong answer left as `none`, an invalid
value, a row from a different build. Run it as often as you like.

## Rules

- **Work alone on the test sheets.** No comparing until both are finished.
- **Never open `_mapping_DO_NOT_SHARE.csv`.** It holds the existing labels and the blind-row list.
- Use `unsure` freely — for items that *cannot be labelled*, not just hard ones (§2C).
- Save as **CSV UTF-8**, never `.xlsx`.

## Order of work

1. `dev/` and `train_spotcheck/` — fast, mostly verification. Do these first to warm up.
2. `test/` — slow and blind. Both of you, independently.
3. Tell me when both test sheets are done → I produce the disagreement list (Step 4).
