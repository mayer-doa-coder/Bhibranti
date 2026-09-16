# Round 1 — the real annotation (M3)

**Status: all four sets annotated and validated. Not yet merged into the corpus.**

Built by [`src/build_annotation_sheets.py`](../../../src/build_annotation_sheets.py), seed 42,
reproducible. Rules followed: [`docs/ANNOTATION_GUIDELINES.md`](../../../docs/ANNOTATION_GUIDELINES.md)
§2C and Rules 11–15.

## What was done

| Folder | Rows | Who | Result |
|---|---:|---|---|
| `test/tawhid/` | 1,356 | Tawhid | ✅ done — blind, every row typed from scratch |
| `test/shejan/` | 1,356 | Shejan | ✅ done — blind, every row typed from scratch |
| `dev/` | 1,346 | single annotator | ✅ done — pre-fill reviewed, 23% corrected, 135 blanks filled |
| `train_spotcheck/` | 1,252 | single annotator | ✅ done — pre-fill reviewed, 26% corrected, 125 blanks filled |

Finished files are named `*_FINAL.csv`. All originals and pre-fill drafts have been removed —
`*_FINAL.csv` is the only version of each file that exists.

## Verified, not just trusted

- **0 structural errors** on all 5,310 rows (`src/validate_annotation.py`)
- **0 rows** where a sheet's question/answer text does not match the corpus record it claims to
  describe (would indicate a stale or copied sheet)
- **0 item_ids** shared between test, dev, and train_spotcheck (no split leakage)
- **0 unsure/unreadable rows missing a note** — every one of the 191 has an explanation
- Pre-filled rows were genuinely reviewed, not accepted blindly: 23–26% were corrected on dev/train

## The test-split result

**Cohen's κ = 0.865** between Tawhid and Shejan, measured on the full 1,356-item test split (not
a sample) — well above the 0.60 floor and higher than the original 100-item pilot (κ = 0.717).

```bash
python src/score_test_agreement.py \
  --a-dir data/annotated/round1/test/tawhid \
  --b-dir data/annotated/round1/test/shejan \
  --a-name Tawhid --b-name Shejan
```

κ is computed on the **1,270** items both annotators could label; 86 more were marked `unsure` or
`unreadable` by at least one of them and are excluded, as the guidelines require. Of the 1,270,
**86 (6.8%)** are genuine correct-vs-wrong disagreements, listed in `test_disagreements.csv`.
(The two 86s are a coincidence — they are different items.)

## Merging into the corpus — `src/merge_annotation.py`

```bash
python src/merge_annotation.py --dry-run            # report only
python src/merge_annotation.py                      # write (refuses while adjudication is pending)
python src/audit.py --data data/splits              # always, after writing
```

It reads every `*_FINAL.csv` here through `_mapping_DO_NOT_SHARE.csv` and **never changes a label
or a difficulty**. A wrong answer gets a type only when the humans settled it:

| Split | Settled automatically | Rule | Needs adjudication |
|---|---:|---|---:|
| test | 487 / 678 | both annotators said `wrong` **and** chose the same type | 191 |
| dev | 648 / 673 | the annotator said `wrong` and chose a type | 25 |
| train | 589 / 3,129 | same, on the 20% spot-check | 37 (+2,503 never sampled) |

Every run writes two files here:

| File | What it is | What you do |
|---|---|---|
| `adjudication.csv` | the 253 wrong answers whose type the humans did not settle (types differ, someone said `correct`, someone was `unsure`) | put a type, `skip`, or `dispute` in `your_decision`. Kept across reruns |
| `label_disputes.csv` | 222 records where at least one human disagrees with the corpus label (116 where every annotator does) | nothing is applied — report it as a corpus finding |

## What is NOT done yet

1. **Adjudication** — 253 rows in `adjudication.csv` have no decision yet.
2. **Running the merge for real** — the dry run is verified; `corpus.jsonl` still says `unlabeled`.
3. **Train coverage decision** — 2,503 train wrong answers were never in the 20% sample. The plan
   (guide §5.3) says to widen *label* verification only if the sample shows > 5% label noise; it
   shows 3.3%. Types for those rows stay `unlabeled` unless you annotate more or accept LLM-proposed
   types (`--with-llm`, marked `llm_consensus`).
4. **Re-running the shortcut gate** after the merge.
5. **Observation:** `overclaim` was never chosen — every typed no-context wrong answer is
   `fabricated`. Worth one sentence in the report.

## Filling a row (for reference)

| Column | What to put |
|---|---|
| `your_label` | `correct` · `wrong` · `unsure` · `unreadable` |
| `your_type` | `none` if correct. If wrong **and there is a passage**: `entity` · `numeric` · `relational` · `contradiction`. If wrong **and there is no passage**: `fabricated` · `overclaim` |
| `your_difficulty` | `easy` · `hard` |
| `your_notes` | a short reason whenever you write `unsure` |
