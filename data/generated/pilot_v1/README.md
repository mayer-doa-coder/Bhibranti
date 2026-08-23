# Pilot v1 — 500-pair Banglish draft (M1 gate input)

**Status: DRAFT. Not reviewed by a human yet. Do not treat as final data.**

Built by [`src/build_pilot.py`](../../../src/build_pilot.py), seed 42, fully reproducible —
running it again produces byte-identical output.

```bash
python src/build_pilot.py
```

## What this is

500 pairs (1,000 records: 300 has-context + 200 no-context, 60/40 per PRD D2) selected from
`data/interim/bn_pool.jsonl` and mechanically converted from Bengali script to a **Banglish
rough draft** — Bangla words, spelled with English letters. This is the pilot the PRD's M1
gate requires (§5.5 V2, §8): *"500 items generated; shortcut probe < 0.60; prompt finalised."*

**This is the input to Task 3 (your manual review), not the output of it.** Nothing here counts
as `human_written` or as annotated until you've read it.

## Files in this folder

| File | Purpose |
|---|---|
| `pilot.jsonl` | The 1,000 records, schema-conformant. What `src/audit.py` reads. |
| **`review_sheet.csv`** | **Open this one to review.** 500 rows, one per pair, Bangla and Banglish side by side. Open in Excel/Sheets. |
| `selection_log.csv` | Which pairs were picked and why (overlap scores, subject). |
| `validation_report.txt` | Counts and the confidence breakdown from the build run. |

## How the 500 pairs were picked

From 17,050 complete pairs in the interim pool, has-context pairs were filtered to remove
exactly the problems `data/DATASET_AUDIT.md` found:

| Filter | Removes | Count excluded |
|---|---|---|
| Fill-in-the-blank | Question repeats the passage with a blank to fill | 776 |
| Passage repetition | Question's own words overlap >50% with the context | 532 |
| Verbatim wrong answer | Hallucinated answer is a direct substring of the context (the 0.832 shortcut) | 183 |
| Low vocabulary reuse | Wrong answer doesn't reuse the passage's words at all (`< 30%` overlap) | 711 |

1,391 pairs survived. The 300 with the **highest** wrong-answer/context word overlap were kept
— these are the pairs where the wrong answer reuses the passage's own vocabulary but gets the
relationship or number wrong, exactly as requested. No subject may exceed 30% of the 300 (a cap,
so one topic can't dominate).

The 200 no-context pairs were filtered for fill-in-the-blank only (context doesn't apply), then
sampled with the same 30% subject cap, seeded for reproducibility.

**Result — verified by re-running the actual audit on this file:**

| Check | Original benchmark | This pilot |
|---|---:|---:|
| Metadata-only shortcut probe (the M1 gate) | 0.506 | **0.505 — PASSES** |
| Context-substring rule (has-context) | 0.832 | **0.742** — improved, not eliminated |

The substring rule dropping to 0.742 is expected, not a bug: a correct grounded answer often
*legitimately* echoes the passage verbatim (that's what makes it correct), so the number can't
go to zero without also making correct answers stop reading naturally. What the filtering
removed is the *wrong* answers being verbatim copies — that was the actual shortcut.

## How the Banglish draft was produced

Two layers, both fully deterministic (same seed, same output every time):

1. **A ~90-word dictionary** of this corpus's own most frequent words and common proper nouns
   (বাংলাদেশ → Bangladesh, ঢাকা → Dhaka, দিবস → dibosh, etc.), each with up to two natural
   spelling variants. A word is picked between variants at random (seeded) per occurrence, which
   is where the "spelling varies on purpose" comes from — e.g. "ebong" vs "ebng" for এবং.
2. **A grapheme fallback** for every other word: every one of the 74 Bengali Unicode characters
   used anywhere in this corpus is mapped to a Latin sound, character by character. This is what
   guarantees **zero untransliterated Bengali survives anywhere** — the build script checks this
   for every field of every record and refuses to write output if it finds any.

### What this method is good at, and what it isn't

Good at: function words, digits, punctuation, and the ~90 dictionary words are read naturally.
Numbers convert perfectly (১৭৭৪ → 1774).

Not good at, by design — and this is exactly what your review is for:
- **English loanwords written in Bengali script** come out wrong. `অব` (the English word "of",
  written phonetically) becomes `obo`, because the fallback has no way to know it's a borrowed
  English word rather than a Bengali one. Watch for this — it's the single most common real
  error type you'll see.
- **Word-final vowel sound.** Real spoken/typed Bengali often drops it (`kotha` vs `kothao`);
  this script always keeps it, because guessing when to drop it would be non-reproducible.
  Rewrite these to how you'd actually type them.
- Every content word not in the small dictionary went through the fallback, so **every single
  record is flagged `transliteration_confidence: "low"`** — read that as "check the content
  words here," not as a defect specific to that record. It's uniform because the dictionary is
  small on purpose; growing it is future work, not required for this pilot.

## What "no error" means here, concretely

Two different kinds of correctness, and only one of them can be guaranteed by a script:

| Guaranteed by the build script (verified, not just claimed) | Needs your human judgement |
|---|---|
| Every record matches `configs/schema.json` | Does the Banglish read naturally? |
| Every pair has exactly one correct + one hallucinated answer | Is `obo` actually "of"? |
| No duplicate ids | Is the hallucination type guess right? |
| Zero leftover Bengali characters in any field | Would a person actually type it this way? |
| Every record traces back to its exact source line | |

The build run confirmed all five left-column checks at 1,000/1,000 records, 0 problems — see
`validation_report.txt`. The right column is Task 3, and it can't be automated away; that's the
whole reason PRD D6 exists.

## How to review (`review_sheet.csv`)

One row per pair. For each row:

1. Read `bn_context` / `bn_question` / `bn_correct_answer` / `bn_wrong_answer` (the originals).
2. Read the matching `banglish_*` column next to it.
3. Fix the Banglish so it reads like something you'd actually type. Edit **`pilot.jsonl`**
   directly for the fields you change (the CSV is for reading and note-taking, not the record of
   truth) — or tell me the changes and I'll apply them.
4. Fill `reviewer_status` (`ok` / `fixed` / `reject`) and `reviewer_notes` for anything unusual.
5. Check `hallucination_type_guess` — it's a heuristic (`numeric` if the wrong answer contains a
   digit, else `relational` for has-context, `fabricated` for no-context). PRD D8 requires a
   human to confirm or correct this; it is not a real annotation yet.

## After review

1. Re-run the gate on your reviewed file:
   ```bash
   python src/audit.py --data data/generated/pilot_v1/pilot.jsonl
   ```
   It must still print `GATE V1 PASSED`. Editing spelling shouldn't move the metadata probe, but
   confirm it anyway.
2. Update `provenance` from `transliterated` to `human_written` for any record you substantially
   rewrote rather than lightly edited — that's what actually counts toward PRD D6's ≥500 target.
3. Tell me when this is done. That closes M1, and unblocks M2 (writing
   `docs/ANNOTATION_GUIDELINES.md`) and scaling from this pilot up to the full ~4,000-record
   corpus.
