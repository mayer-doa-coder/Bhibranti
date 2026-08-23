# `data/annotated/` — human-labelled data

## `agreement_test_v1/` — do this first

The M2 gate: two people label the same 100 blind items using
[`docs/ANNOTATION_GUIDELINES.md`](../../docs/ANNOTATION_GUIDELINES.md), and Cohen's kappa between
them must reach 0.60. See [`agreement_test_v1/README.md`](agreement_test_v1/README.md).

Nothing else in this folder exists yet.

## What goes here later

Once M2 passes, the labelled corpus lands here — the same records as
`data/corpus/bn_v1/corpus.jsonl` but with `hallucination_type`, `annotator_1`, `annotator_2`,
and `adjudicated` filled in by people instead of left `unlabeled`.

Note `difficulty` is **already** set (easy/hard) by `src/build_corpus.py` — it records whether
the string shortcut separates that pair. Annotators also judge easy/hard independently; the two
are different signals and both are worth having.

Annotation plan (PRD): test split gets both annotators plus adjudication, dev gets one annotator,
train gets a 20% spot-check.
