# Dataset audit — `bn_qa_pool` (the supplied "small dataset" files)

**Audited:** 23 August 2026 · **Records examined:** 62,084 across 14 files
**Reproduce:** `python src/build_bn_pool.py && python src/audit.py --data data/splits/bn_dev_benchmark`

---

## Verdict

> **Not sufficient as the Phase 1 corpus. Genuinely valuable as the base QA layer it is
> built from.**

The blocker is not size or quality — it is **language variety**. This pool is monolingual
**Bengali script**; 0.96% of its alphabetic characters are Latin. BanglishHallu is about
**romanised Bangla–English code-mixed text**. There is essentially no Banglish in here.

That is a recoverable position, not a dead end. The implementation guide (§3.3) recommends
BEnQA precisely *because* it is Bengali-script, so that producing the Banglish condition is a
**transliteration** step rather than a translation step. This pool fits the same slot, and it is
substantially larger than BEnQA (~5K): after cleaning it yields **19,420 complete
faithful/hallucinated pairs** — nearly 5× the PRD's 4,000-record target.

So: this is `data/raw/`, not `data/splits/`. It removes the *sourcing* problem (M0) entirely.
It does not remove the *Banglish generation* problem (M1) or the *annotation* problem (M2–M3).

---

## What the data is

| Property | Value |
|---|---|
| Files | 14 `.jsonl`, one per subject |
| Records | 62,084 |
| Content sources | Bengali Wikipedia (CC BY-SA 4.0) + BCS question banks |
| How built | QA pairs constructed with LLM assistance from those source texts |
| Schema | `{context, prompt_bn, response_bn, label}` — 100% consistent, zero parse errors |
| Class balance | 49.7% / 50.3% — excellent |
| has-context | 12,348 (19.9%) |
| no-context | 49,736 (80.1%), marked `context: "[NULL]"` |
| Complete label pairs | 19,439 groups (same context+question, both labels) |
| Unpaired records | 12,936 |
| Response length | mean 2.85 tokens, **median 1**, 64% are a single token |

Subjects: mathematics (29,332), grammar (5,000), history (5,000), reading_comprehension (5,000),
science (4,000), bcs (2,324), antonym (2,000), idiom_meaning (2,000), vocabulary (2,000),
geography (1,741), law (1,383), synonym (1,000), literature (914), others (390).

---

## Finding 1 — Label polarity established: `1 = correct`

**The source files encode `1 = correct`, `0 = hallucinated`.** The project has adopted this as
its convention (PRD §5.1a), so **no flip is applied anywhere in the pipeline**.

Establishing this empirically mattered: the files carry no documentation, and reading the
polarity backwards would train every model in the ladder inverted while producing
normal-looking metrics — macro-F1 is symmetric under a global label flip, so nothing would look
wrong until a per-class confusion matrix was read.

Confirmed three independent ways:

**a. Context overlap** (has-context records, where the faithful answer should be grounded):

| `label` | n | Mean token overlap with context | Answer appears verbatim in context |
|---|---:|---:|---:|
| `1` (correct) | 6,297 | **0.765** | **78.6%** |
| `0` (hallucinated) | 6,051 | 0.484 | 6.9% |

**b. Externally checkable facts:**

| `label` | Question | Answer | Truth |
|---|---|---|---|
| `1` | বাংলাদেশের জাতীয় দিবস কোনটি? | ২৬ মার্চ | ✅ correct |
| `0` | বাংলাদেশের জাতীয় দিবস কোনটি? | ১৬ ডিসেম্বর | ❌ that is Victory Day |
| `1` | বাংলাদেশের বিজয় দিবস কবে? | ১৬ ডিসেম্বর। | ✅ correct |
| `0` | বাংলাদেশের বিজয় দিবস কবে? | ১৪ ডিসেম্বর। | ❌ |

**c. Domain checks:** "নিচের কোন সংখ্যাটি মৌলিক?" → `৪৭` (prime) is `label 1`;
`৮৭` (= 3 × 29) is `label 0`.

So `২৬ মার্চ / label 1` is the **correct** answer and `১৬ ডিসেম্বর / label 0` is the
hallucinated one — consistent with the project convention throughout.

---

## Finding 2 — This is Bengali script, not Banglish (blocking, unresolved)

| Measure | Value |
|---|---|
| Bengali-script characters | 9,545,106 |
| Latin-script characters | 92,908 — **0.96%** of alphabetic |
| Records containing any Latin letter | 4,808 (7.7%) |
| Records >30% Latin (plausibly code-mixed) | 1,027 (**1.7%**) |

The Latin that exists is mostly parenthetical glosses (`দ্য রিপাবলিক (The Republic)`), not
code-mixing. The `vocabulary` file is the only one above 3% Latin (14.3%), because it glosses
English meanings.

**Consequence:** the project's central claim — that hallucination detection degrades on
Bangla–English code-mixed text — cannot be evaluated on this data at all. BanglishBERT and
MuRIL are in the model ladder specifically because they saw *transliterated* text during
pretraining; on pure Bengali script that advantage disappears and BanglaBERT would likely win,
which measures something else entirely.

**What closes the gap** (guide §3.3, all three routes):
1. Native writers rewrite items as they would actually type them — target ≥ 500 items (PRD D6).
2. Transliteration tooling (`bnbphoneticparser`, `indic_transliteration`, `bntranslit`) over the
   Bengali-script version, with a native speaker spot-checking ~10%.
3. LLM rewriting instructed to preserve natural spelling variance.

This pool is an excellent input to all three: the question/answer pairs already exist and are
already label-balanced, so only the surface form has to change.

---

## Finding 3 — The metadata shortcut gate PASSES

The PRD's blocking gate (V1, guide §9.1) — logistic regression on content-free surface features
of the candidate answer alone:

| Probe | Split | macro-F1 | Verdict |
|---|---|---:|---|
| Metadata-only | random | 0.453 | **CLEAN** (< 0.55) |
| Metadata-only | grouped | 0.494 | **CLEAN** |
| Metadata-only | benchmark splits | 0.506 | **CLEAN** |
| Answer-only char n-gram | grouped | 0.581 | mild lexical signal |
| Majority class | — | 0.338 | as expected |

Length parity is good on its own terms: 2.77 tokens (hallucinated) vs 2.93 (faithful). The
short answers make length control almost automatic.

**This is real good news, and now it means something specific.** The pool *was* LLM-constructed
(see `data/SOURCES.md`), which is exactly the condition RK1 warns about — yet the classic
generation artifact is absent. The length-parity and surface-feature discipline held. It is the
one Phase 1 risk rated "Critical" that this data does *not* carry.

---

## Finding 4 — But there is a different shortcut, and it is severe

A probe the PRD does not specify, added in `src/audit.py`:

```
TRIVIAL RULE: "candidate answer is a substring of the context" -> faithful
  macro-F1 on has-context items = 0.832
```

**No learning at all — pure string matching — beats the project's has-context target of 0.80.**

Faithful answers appear verbatim in the context 78.6% of the time; hallucinated ones 6.9%. The
has-context split is therefore mostly an **extractive span-matching** task, not a faithfulness
task. Any encoder will score high on it and the score will mean very little.

This also puts PRD **V5** in play pre-emptively: a > 0.95 has-context result on this data should
be assumed to be this shortcut, not detection.

**Mitigation for the real corpus:** generate hallucinated answers that also draw their surface
tokens from the context (the `relational` and `contradiction` types do this naturally — correct
entities, wrong relation), and generate faithful answers that paraphrase rather than copy. The
guide's paired-generation protocol (§4.1) produces this if the prompt asks for it; this pool's
generator evidently did not.

---

## Finding 5 — Structural issues (all corrected in `interim/`)

| Issue | Count | Handling |
|---|---:|---|
| Exact duplicate records | 5,536 (8.9%) | dropped |
| Contradictory records (identical text, both labels) | 68 | dropped |
| Records surviving cleaning | **56,480** | → `data/interim/bn_pool.jsonl` |
| Unique responses | 26,599 / 62,084 | high reuse; grouped splits are mandatory |

**Subject imbalance:** mathematics alone is 10,007 of 19,420 complete pairs (51.5%), all
no-context. Left uncorrected, the benchmark would largely measure arithmetic verification. The
builder caps any subject at 25% of its condition.

---

## Finding 5a — The corpus is LLM-constructed, which affects the PRD's positioning

The content comes from Bengali Wikipedia passages and BCS question banks; the paired correct /
incorrect answers were built from those texts with LLM assistance. The facts are sourced, not
invented — but the *items* are synthetic in construction.

PRD §2.2 positions this project against BenHalluEval on the grounds that BenHalluEval's
code-mixed track is synthetically generated while this corpus is **naturalistic**. On the
current pool that distinction does not yet hold.

It is restored by two things already in the plan, and they now carry more weight than their
"Should" priority suggests:

- **D6** — ≥ 500 items with human-written Banglish from native writers. This is the actual
  differentiator.
- **D7** — 100–200 human-written hallucinated answers held out in test, so the generated-vs-human
  gap (V4) is measured rather than assumed.

Every record now carries `provenance` so the two populations stay separable. Keep it accurate —
Phase 2's headline claim about synthetic-vs-natural transfer depends entirely on it.

---

## Finding 6 — Fields the project needs that do not exist

| Schema field | Status |
|---|---|
| `hallucination_type` | **Absent.** PRD D8 requires the 6-type taxonomy on every positive. Set to `unlabeled`. |
| `difficulty` | **Absent.** PRD D9 wants ~60/40 easy/hard. Set to `unlabeled`. |
| `annotator_1` / `annotator_2` / `adjudicated` | **Absent.** No human annotation → PRD D4, D5 unmet, κ cannot be reported. |
| `level` (SSC/HSC/BCS) | Absent except by inference from the `bcs` file. Set to `unknown`. |
| `source` / licence | **Resolved.** Bengali Wikipedia (CC BY-SA 4.0) + BCS question banks. Two open questions remain: BCS compilation terms, and the released-corpus licence. See `data/SOURCES.md`. |
| `provenance` | Set to `llm_generated`. Needed so the human-written comparison (D7, V4) stays possible. |
| `script_condition` | Set to `bengali_script` (not `banglish`). |
| `cmi` | Computed as a **script proxy**, ≈ 0 throughout. Must be recomputed with real token-level language ID once Banglish exists. |
| `error_span` | Absent. Phase 2 payload; recoverable only at generation time. |

---

## Scorecard against PRD requirements

| Req | Requirement | Status |
|---|---|---|
| D1 | ≥ 4,000 pairs | ✅ 19,420 complete pairs available |
| D2 | 60/40 has/no-context | ⚠️ pool is 20/80; enough has-context exists to subsample to 60/40 |
| D3 | 50/50 class balance | ✅ 49.7/50.3 |
| D4 | 500-item test set, human-verified, double-annotated | ❌ no annotation |
| D5 | κ ≥ 0.60 | ❌ not computable |
| D6 | ≥ 500 human-written Banglish items | ❌ zero Banglish |
| D8 | Hallucination type on all positives | ❌ absent |
| D9 | Easy/hard difficulty | ❌ absent |
| D10 | Licence recorded for every source | ⚠️ sources known (Wikipedia CC BY-SA 4.0 + BCS banks); BCS terms and release licence still open |
| D11 | `script_condition` + `cmi` populated | ⚠️ populated, but values reflect Bengali script |
| D13 | No PII | ⚠️ not yet scanned |
| V1 | Metadata probe < 0.60 | ✅ **0.453 – 0.506, passes** |
| V3 | Answer-only well below full input | ✅ 0.581 |
| — | Code-mixed Banglish (PRD §7.1 scope) | ❌ **0.96% Latin — the blocking gap** |

**7 of 14 unmet, 1 blocking for the project's premise (Banglish).** D10 moved from unknown to
partly resolved.

---

## What this data unblocks right now

Do not wait on the Banglish work to start building. With `data/splits/bn_dev_benchmark/` you
can, today:

- Build and debug the entire model ladder end-to-end — **milestone M4**, on real data
- Validate `train_classical.py`, `train_transformer.py`, `evaluate.py`, the 5-fold CV harness,
  the 3-seed loop, and the experiment log
- Establish a Bengali-script reference point that becomes a genuinely interesting comparison:
  *the same detector, same questions, Bengali script vs Banglish* is close to Phase 2's headline
  claim and costs nothing extra to have

When reporting any number from this benchmark, state that it is Bengali-script and that the
0.832 substring shortcut is present. It is a pipeline test, not a result.

---

## Recommended next actions

1. **Close the two remaining licence questions** (`data/SOURCES.md`): whether the BCS questions
   came from an official PSC source or a commercial compilation, and what licence the released
   corpus carries. Wikipedia's CC BY-SA 4.0 share-alike almost certainly propagates to it.
2. **Run the Banglish generation step** (guide §3.3) over the cleaned pool — this is the actual
   critical path.
3. **Fix the copy-from-context shortcut in the generation prompt** before generating at scale,
   or the real corpus inherits Finding 4.
4. **Write `docs/ANNOTATION_GUIDELINES.md`** and recruit annotators (M2). This is the longest
   lead-time item and is currently at zero.
5. **PII scan** the pool (D13) before anything is published.
6. Keep `data/splits/train|dev|test.jsonl` — the real, locked Phase 1 splits — **empty until
   M3**. The benchmark under `bn_dev_benchmark/` must never be promoted into them.
