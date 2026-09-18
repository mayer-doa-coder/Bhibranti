# Source and licence register

PRD **D10 is a Must-priority requirement**: every data source must have its licence recorded
before ingestion, and the licence must be carried forward into the release. Reviewers and
journal editors will ask.

**Status: sources identified. Two licence questions remain open — see the actions at the bottom.**

> **Phase 1 is Bengali-first as of 2026-08-23.** The Banglish sources listed under "Planned"
> below are **Phase 2** material and must not be ingested yet. The Phase 1 corpus is
> `data/corpus/bn_v1/corpus.jsonl`, derived only from the pool registered immediately below.

---

## Ingested

### `bn_qa_pool` — the Bengali-script QA pool

| Field | Value |
|---|---|
| Location | `data/raw/bn_qa_pool/` (14 `.jsonl` files) |
| Records | 62,084 (56,480 after cleaning) |
| Language | Bengali script (0.96% Latin characters) |
| Domain | Educational — BCS/SSC/HSC style, plus grammar, vocabulary, mathematics |
| Compiled by | Tawhidul Hasan, July 2026 |
| Label polarity | `1` = correct, `0` = hallucinated (matches PRD §5.1a; no flip applied) |

#### Where the content comes from

| # | Source | What it supplied | Licence | Redistributable |
|---|---|---|---|---|
| 1 | **Bengali Wikipedia** (bn.wikipedia.org) | Passage text, collected manually into plain-text files, then used as the grounding context for the has-context items | **CC BY-SA 4.0** | ✅ Yes — **with attribution and share-alike** |
| 2 | **BCS question banks** | Exam questions and answer options, chiefly the no-context items | ❓ **Check publisher terms** | ❓ Unresolved |

#### How the records were built

The question/answer pairs — both the correct and the incorrect answer for each item — were
**constructed with LLM assistance (Claude) from the source texts above**, not written by hand
and not drawn from Claude's own knowledge. The facts originate in the Wikipedia passages and
the BCS banks; the LLM was used to turn that source text into paired QA items.

This is recorded because the PRD depends on it in three places, not as a caveat about quality:

- **§2.2 positioning.** The project's stated advantage over BenHalluEval is that this corpus is
  *naturalistic, not synthetic*. As it stands, the pool is synthetic in the same sense theirs
  is. See "Impact on the naturalistic claim" below.
- **§14 Phase 2.** The headline claim is that detectors trained on synthetic code-mixing
  overestimate performance on natural Banglish. Testing that requires knowing which items are
  which.
- **§5.5 V4 / guide §9.2.** The human-written holdout exists precisely to measure the gap
  against generated items. Without a generated/human split recorded, the check is meaningless.

**Still to record** (PRD R3, and the `generator_model` schema field):

- [ ] Exact Claude model and version used
- [ ] The prompt(s) used, committed to the repo
- [ ] Roughly how many items came from route 1 (Wikipedia) vs route 2 (BCS banks)

#### Licence consequences

**Wikipedia — CC BY-SA 4.0.** This is the binding constraint on release:

- **Attribution** required — credit Bengali Wikipedia and link the licence.
- **Share-alike** — a derivative of CC BY-SA content must be released under CC BY-SA 4.0. Your
  released corpus (DL1) very likely inherits this. Plan for it now; do not promise a permissive
  licence you cannot grant.
- Compatible with `csebuetnlp` sources? **No.** Those are CC BY-**NC**-SA 4.0 (non-commercial).
  Mixing CC BY-SA and CC BY-NC-SA into one released file creates a licence conflict. If you
  later ingest `squad_bn`, keep it in a separate file with its own licence note.

**BCS question banks — unresolved.** Public exam questions are often reusable, but a
*published compilation* (a guide book, a coaching centre's bank, a website's collection) is
usually copyrighted as a compilation even when the individual questions are not. What you need
to check:

- Did the questions come from an official PSC source, or from a commercial guide/website?
- If commercial: treat as **internal benchmarking only** until cleared, and do not ship those
  items in the released corpus.

#### Impact on the "naturalistic" claim (PRD §2.2)

The PRD distinguishes this work from BenHalluEval on the grounds that BenHalluEval's code-mixed
track is *synthetically generated* while this corpus is *naturalistic*. On the current pool that
distinction does not yet hold — the items are LLM-constructed from source text.

Two ways to restore the claim, both already in the plan:

1. **PRD D6** — ≥ 500 items with **human-written** Banglish, produced by native writers. This is
   what makes the corpus naturalistic, and it becomes the real differentiator.
2. **PRD D7** — 100–200 **human-written hallucinated answers** held out in the test set, so the
   generated-vs-human gap is measured rather than assumed.

Until those exist, describe the corpus honestly as LLM-constructed over Wikipedia and BCS
sources, with a human-written subset planned. Set `provenance` per record so the two are always
separable.

---

## Planned (not yet ingested)

| Source | Licence | Notes |
|---|---|---|
| **BEnQA** — `github.com/sheikhshafayat/BEnQA` | check repo | ~5K parallel Bengali/English SSC & HSC science questions |
| **NCTB-QA** — arXiv 2603.05462 | check paper | 87,805 QA pairs, 42.75% unanswerable — the `overclaim` type lives here |
| **BanglaRQA** | public, check terms | 14,889 QA pairs |
| **squad_bn** — `csebuetnlp/squad_bn` | **CC BY-NC-SA 4.0** | Non-commercial. **Licence-incompatible with CC BY-SA Wikipedia content — keep separate.** |
| **BanglaTLit-PT** — Kaggle | check dataset page | 243K unlabeled transliterated Bangla; the further-pretraining corpus |
| **BanglishRev** — arXiv 2412.13161 | check paper | 1.74M Bangla/English/Banglish e-commerce reviews |
| **MixSarc** — `ajwad-abrar/MixSarc` | check HF card | Naturally-occurring Banglish from Facebook |
| **অলীকবচন Kaggle** — `kaggle.com/competitions/bengali-hallucination` | **read the rules tab** | PRD Q1. A Bengali LLM hallucination-detection competition; a plausible origin of part of this pool. Competition licences often permit competition use only. **Resolve before releasing.** |
| **BenHalluEval** — arXiv 2605.31483 | check paper | The Bengali hallucination benchmark (12,000 hallucinated candidates, four tasks). Reference point and related work — not a source to ingest. |
| **TyDiQA-GoldP (Bengali)** | Apache 2.0 (check) | Grounded Bengali QA. The best candidate if has-context data must be scaled — see guide §4. |

## Models

| Model | Licence |
|---|---|
| `google/muril-base-cased` | Apache 2.0 |
| `csebuetnlp/banglishbert`, `csebuetnlp/banglabert` | check model card |
| `xlm-roberta-base`, `bert-base-multilingual-cased` | MIT / Apache 2.0 |
| `ai4bharat/IndicBERTv2-MLM-only` | check model card |

## Language resources

| Resource | Where it is used | Source | Licence |
|---|---|---|---|
| Bengali stop-word list, 398 entries, copied unchanged | `configs/bn_stopwords.txt` — M10 ablation only (`src/text_bn.py`) | stopwords-iso, "bn" list, Python package `stopwordsiso` 0.7.1 (github.com/stopwords-iso/stopwords-iso), downloaded 2026-09-17 | MIT |
| Protected words, suffix list | `configs/bn_protected_words.txt`, `configs/bn_suffixes.txt` | written for this project | project licence |

---

## Open actions

| # | Action | Blocks |
|---|---|---|
| 1 | Confirm where the BCS questions came from (official PSC vs commercial compilation) | Release of those items |
| 2 | Decide and state the released corpus licence — CC BY-SA 4.0 is the likely inherited answer | DL1, DL4 |
| 3 | Record the Claude model/version and commit the construction prompts | PRD R3 |
| 4 | Add a Wikipedia attribution file listing the articles used | CC BY-SA compliance |
| 5 | PII scan before publication | PRD D13 |

---

## Register template

Copy this block for each new source, and fill it **before** the data enters `data/raw/`.

```
### <name>
| Field | Value |
|---|---|
| Location | data/raw/<dir>/ |
| Upstream URL | |
| Retrieved | YYYY-MM-DD |
| Licence | |
| Redistributable | yes / no / research-only |
| Citation | |
| Used for | |
```
