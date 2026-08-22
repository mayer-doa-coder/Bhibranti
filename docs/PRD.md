# Product Requirements Document — BanglishHallu Phase 1

**Project:** BanglishHallu — Bangla–English Code-Mixed Hallucination Detection
**Phase:** 1 of 2 — Detector Construction & Score Maximisation
**Document version:** 1.0
**Date:** 23 August 2026
**Status:** Draft for approval

---

## 1. Executive summary

Phase 1 delivers a **working Bangla–English code-mixed hallucination detector** with a validated, human-annotated corpus and a benchmarked model ladder. The immediate objective is demonstrable detection performance; the strategic objective is to produce a corpus and pipeline that Phase 2 can convert into a publishable research contribution.

Phase 1 is deliberately narrow. It does not attempt the causal analysis of how code-mixing degrades detection — that is Phase 2's contribution and requires this phase's infrastructure to exist first.

---

## 2. Background and problem statement

### 2.1 The problem

Large language models hallucinate — producing fluent, confident output that contradicts the provided context or fabricates facts. Detection of these hallucinations is an established research area for English and, increasingly, for medium-resource languages.

Bangla–English code-mixed text ("Banglish") is how a large fraction of Bangladeshi users actually interact with digital systems. It combines two difficulties: Bangla is low-resource, and romanised Bangla has no orthographic standard, so the same word appears with many spellings. Models pretrained on either standard Bangla or English handle it poorly. Prior code-mixed Bangla work found transliteration-induced out-of-vocabulary tokens in roughly 39% of a dataset, which is catastrophic for models never pretrained on such text.

No public, human-annotated, naturalistic Bangla–English code-mixed hallucination **detection** corpus exists.

### 2.2 Competitive landscape

| Work | What it covers | Gap it leaves |
|---|---|---|
| **BenHalluEval** (arXiv 2605.31483) | Bengali hallucination benchmark; 12,000 candidates, 12 types, 4 tasks incl. Bangla–English Code-Mixed QA | Code-mixed track is **synthetically generated** — LLM-converted from 1,000 Bengali GQA seeds. It is an LLM-as-judge *evaluation* framework, not a trainable detection corpus, and does no human annotation of the code-mixed data. |
| **SHROOM-CAP 2025** | Multilingual scientific hallucination, 9 languages, Bengali as zero-shot | Bengali is zero-shot only; not code-mixed; scientific domain |
| **BanTH** (arXiv 2410.13281) | 37.3K transliterated Bangla, hate speech | Different task; provides the methodology template |
| **MixSarc** (arXiv 2602.21608) | Naturalistic Banglish, sarcasm/humour | Different task |
| **ViHallu** (arXiv 2601.04711) | Vietnamese hallucination, 10K triplets, 111 teams | Different language; provides the target paper structure |

**Positioning:** Phase 1 builds the artifact that occupies the gap — naturalistic (not synthetic) code-mixing, human-annotated, with both grounded and closed-book conditions.

### 2.3 Why now

The BenHalluEval group is actively publishing in this exact space (the same authors also produced MixSarc). The window for a naturalistic code-mixed hallucination resource is open but not indefinitely.

---

## 3. Goals and non-goals

### 3.1 Goals (Phase 1)

| ID | Goal |
|---|---|
| G1 | Construct a human-annotated Bangla–English code-mixed hallucination corpus of ≥ 4,000 QA pairs |
| G2 | Cover both grounded (has-context) and closed-book (no-context) conditions |
| G3 | Benchmark a full model ladder from lexical to pretrained-contextual |
| G4 | Achieve macro-F1 ≥ 0.80 on has-context and ≥ 0.60 on no-context |
| G5 | Demonstrate corpus validity via a passing shortcut audit |
| G6 | Produce a reproducible pipeline and complete experiment log |

### 3.2 Non-goals (deferred to Phase 2)

| ID | Deferred item | Rationale |
|---|---|---|
| N1 | Script-controlled CMI degradation study | Requires the parallel four-condition design; Phase 1 records the fields but doesn't run the experiment |
| N2 | Transliteration normalisation as a *research finding* | Phase 1 uses it as a preprocessing arm only |
| N3 | Natural vs. synthetic transfer experiment | Needs a synthetic comparison corpus |
| N4 | Span-level hallucination annotation | Phase 1 captures `error_span` opportunistically but doesn't evaluate on it |
| N5 | Tokenizer fertility analysis | Phase 2 mechanism analysis |
| N6 | LLM fine-tuning, RLHF, retrieval augmentation | Out of scope entirely |
| N7 | Paper writing and venue submission | Phase 2 |

### 3.3 Explicit anti-goals

- **Do not maximise score at the expense of corpus validity.** A model scoring 0.97 on an artifacted corpus is a failed deliverable.
- **Do not chase state-of-the-art architectures.** The model ladder is fixed by pedagogical requirement and is scientifically defensible as-is.

---

## 4. Stakeholders

| Stakeholder | Interest | Success looks like |
|---|---|---|
| Course instructor | Demonstrated command of taught methods (N-gram, Skip-gram, RNN, LSTM, BERT) applied to a real problem | Full ladder implemented, results explained, mechanism understood |
| Student / researcher | Grade; foundation for publication | G1–G6 met; corpus reusable in Phase 2 |
| Phase 2 (future self) | A clean, extensible corpus and pipeline | Schema includes Phase 2 fields; test set never contaminated |
| Research community | A usable low-resource resource | Public release with licence, guidelines, IAA |

---

## 5. Requirements

### 5.1 Data requirements

| ID | Requirement | Priority | Acceptance |
|---|---|---|---|
| D1 | ≥ 4,000 annotated QA pairs | **Must** | Count in `data/splits/` |
| D2 | 60/40 has-context / no-context split | Must | Verified by script |
| D3 | 50/50 class balance (hallucinated/faithful) | Must | ±5% tolerance |
| D4 | 500-item test set, 100% human-verified, double-annotated | **Must** | Adjudication log exists |
| D5 | Cohen's/Fleiss' κ ≥ 0.60 on binary label | **Must** | Reported in `results/` |
| D6 | ≥ 500 items with human-written (not generated) Banglish | Should | Provenance field populated |
| D7 | 100–200 human-written hallucinated answers held out in test | Should | Flagged in test set |
| D8 | Hallucination type labelled for all positive instances | Must | 6-type taxonomy (§5.2) |
| D9 | Easy/hard difficulty labelled, ~60/40 | Should | Field populated |
| D10 | Licence recorded for every source | **Must** | `data/SOURCES.md` complete |
| D11 | `script_condition` and `cmi` fields populated | Should | Phase 2 enabler |
| D12 | `error_span` captured where available | Could | Phase 2 enabler |
| D13 | No PII in released data | **Must** | Scrub + manual review |

### 5.2 Taxonomy requirement

The corpus must use the field's existing intrinsic/extrinsic framing (ViHallu, HalluLens), not a bespoke scheme:

**Intrinsic (has-context):** `entity`, `numeric`, `relational`, `contradiction`
**Extrinsic (no-context):** `fabricated`, `overclaim`
**Negative class:** `none`

### 5.3 Model requirements

| ID | Requirement | Priority |
|---|---|---|
| M1 | N-gram baselines (LogReg + SVM), word and character features | **Must** — pedagogical |
| M2 | Skip-gram embedding baselines (LogReg + XGBoost) | **Must** — pedagogical |
| M3 | BiRNN and BiLSTM (+ attention variant) | **Must** — pedagogical |
| M4 | ≥ 5 pretrained transformer encoders fine-tuned | **Must** |
| M5 | BanglishBERT and MuRIL specifically included | **Must** — the two models pretrained on transliterated/bilingual data |
| M6 | Further pretraining on ≥ 1 encoder | Should |
| M7 | Soft-voting ensemble of top 3 | Should |
| M8 | Zero-shot LLM reference point (≥ 1 model, ≥ 300 items) | Should |
| M9 | 3 preprocessing arms × 3 input formats swept on best model | Should |

### 5.4 Evaluation requirements

| ID | Requirement | Priority |
|---|---|---|
| E1 | Macro-F1 as primary metric | **Must** |
| E2 | Results reported separately for has-context / no-context | **Must** |
| E3 | Results reported by difficulty (easy/hard) | Should |
| E4 | Results reported per hallucination type | Should |
| E5 | 5-fold stratified CV on train+dev | **Must** |
| E6 | 3 random seeds, mean ± std reported | **Must** |
| E7 | McNemar's test for pairwise model comparison | Should |
| E8 | Bootstrap 95% CI on headline result | Should |
| E9 | Test set opened exactly once, at the end | **Must** |
| E10 | OOV rate and vocabulary coverage logged for classical models | Should |

### 5.5 Validity requirements — the gate

| ID | Requirement | Priority | Gate |
|---|---|---|---|
| V1 | Metadata-only shortcut probe scores < 0.60 macro-F1 | **Must** | **Blocking** |
| V2 | Shortcut probe run on the 500-item pilot before full generation | **Must** | **Blocking** |
| V3 | Answer-only probe substantially below full-input model on has-context | Should | Investigate if violated |
| V4 | Human-written holdout performance within reasonable range of generated | Should | Report the gap either way |
| V5 | Any macro-F1 > 0.95 triggers a leakage investigation | **Must** | Documented |

**V1 and V2 are release gates.** If the pilot fails V1, generation stops, the prompt is fixed, and the pilot is regenerated. No exceptions — this is the single failure mode that would invalidate the entire project.

### 5.6 Reproducibility requirements

| ID | Requirement | Priority |
|---|---|---|
| R1 | `results/experiment_log.csv` records every run incl. failures | **Must** |
| R2 | All seeds fixed and logged | **Must** |
| R3 | Generation prompts version-controlled | **Must** |
| R4 | Annotation guidelines document written before annotation begins | **Must** |
| R5 | Environment pinned (`requirements.txt`) | Should |
| R6 | Data splits deterministic and script-generated | **Must** |

---

## 6. Success metrics

### 6.1 Primary

| Metric | Floor | Target | Stretch |
|---|---|---|---|
| Has-context macro-F1 | 0.75 | **0.85** | 0.90 |
| No-context macro-F1 | 0.55 | **0.68** | 0.75 |
| Corpus size | 1,500 | **4,000** | 6,000 |
| Inter-annotator κ | 0.60 | **0.70** | 0.75 |
| Shortcut probe | < 0.60 | **< 0.55** | < 0.52 |

### 6.2 Secondary

| Metric | Target |
|---|---|
| Hard-subset macro-F1 | ≥ 0.60 |
| Best-vs-worst encoder spread | Reported with significance test |
| FPT gain over base | ≥ +1.5 points on at least one model |
| Ensemble gain over best single | ≥ +1.0 points |

### 6.3 External calibration

These are the numbers your results will be compared against:

| Benchmark | Task | Best reported |
|---|---|---|
| ViHallu | Vietnamese hallucination, 3-class, w/ context | 84.80 macro-F1 (best of 111 teams); 32.83 encoder baseline |
| BanTH | Transliterated Bangla, binary | 77.36 macro-F1 (further-pretrained mBERT) |
| MedHallu | Medical hallucination, hard subset | 0.625 F1 |
| SHROOM-CAP | Bengali zero-shot factuality | ~0.51 F1 |

A has-context result in the 0.80–0.90 band is competitive and defensible. Results above 0.95 are implausible and must be audited.

---

## 7. Scope boundaries

### 7.1 In scope

- Binary hallucination classification (hallucinated / faithful)
- Bangla–English code-mixed text, primarily romanised (Banglish)
- Educational domain: BCS / SSC / HSC level question answering
- Both grounded and closed-book conditions
- Classical → pretrained-contextual model ladder
- Further pretraining, ensembling, threshold tuning
- Corpus validity auditing

### 7.2 Out of scope

- Multi-class or multi-label hallucination classification
- Span-level detection and evaluation
- Hallucination *mitigation* or correction
- Retrieval-augmented generation
- Other language pairs
- Real-time / production deployment
- Domains beyond education
- LLM fine-tuning

---

## 8. Milestones

| M# | Week | Milestone | Exit criteria |
|---|---|---|---|
| **M0** | 1 | Foundations | Repo, schema, licences audited, seeds fixed, test set designated and locked |
| **M1** | 1 | **Pilot + validity gate** | 500 items generated; **shortcut probe < 0.60**; prompt finalised |
| **M2** | 2 | Annotation protocol | Guidelines written; 100-item pilot annotated; κ computed; guidelines revised |
| **M3** | 3 | Corpus complete | 4,000 items generated + annotated; IAA reported; splits created |
| **M4** | 4 | Pipeline working | Classical ladder + 1 transformer trained end-to-end; log populated |
| **M5** | 5 | Full benchmark | All models × 3 arms × 3 formats; 5-fold CV; 3 seeds |
| **M6** | 6 | **Phase 1 complete** | FPT + ensemble + threshold + LLM reference; all tables; final audit passed |

**M1 is a hard gate.** No progression to M3 without a passing shortcut probe.

---

## 9. Deliverables

| ID | Deliverable | Format |
|---|---|---|
| DL1 | Annotated corpus | JSONL, train/dev/test splits |
| DL2 | Annotation guidelines | Markdown, with worked bilingual examples |
| DL3 | IAA report | κ per label type, adjudication log |
| DL4 | Source and licence register | `data/SOURCES.md` |
| DL5 | Trained model checkpoints | HF format, best model per family |
| DL6 | Results tables 1–5 | Markdown / CSV |
| DL7 | Experiment log | `results/experiment_log.csv`, complete |
| DL8 | Validity audit report | Shortcut probe, answer-only probe, human holdout |
| DL9 | Reproducible codebase | Git repo, pinned deps, deterministic splits |
| DL10 | Phase 1 technical report | 6–10 pages, methods + results + limitations |

---

## 10. Risks

| ID | Risk | L | I | Mitigation | Owner |
|---|---|---|---|---|---|
| **RK1** | **Generation artifacts inflate scores; corpus invalid** | **H** | **Critical** | Shortcut probe as a blocking gate at M1; length control enforced programmatically; identical prompt template for both classes | Student |
| RK2 | IAA below 0.60 | M | High | 100-item pilot; explicit edge-case rules; expert adjudication | Student |
| RK3 | Annotator attrition | M | Med | Recruit 3, require 2 | Student |
| RK4 | Kaggle data licence prohibits use | M | Low | Treat as external benchmark only; don't build the corpus on it | Student |
| RK5 | Model spread too tight to distinguish | M | Med | 3 seeds, McNemar, bootstrap CI; report ties honestly | Student |
| RK6 | No-context split underperforms | H | Low | Expected outcome; report as a finding, not a failure | — |
| RK7 | Competing group publishes first | M | Med | Phase 1 is a course deliverable; Phase 2 positions against, not ahead of, BenHalluEval | Student |
| RK8 | GPU quota exhausted | L | Med | Base models only; `max_length=256`; gradient accumulation | Student |
| RK9 | Scope creep into Phase 2 work | **H** | Med | This PRD's §3.2 is the boundary; defer anything on that list | Student |
| RK10 | Test set contamination via repeated evaluation | M | High | Test set locked at M0; opened once at M6 | Student |

RK9 deserves emphasis. The Phase 2 experiments are more interesting than the Phase 1 ones, and the temptation to start them early is real. Resist it — Phase 2 depends on a validated corpus, and a half-built corpus with half-built experiments delivers neither.

---

## 11. Dependencies

### 11.1 External

| Dependency | Type | Risk if unavailable |
|---|---|---|
| BEnQA | Base QA source | Low — NCTB-QA, BanglaRQA are substitutes |
| BanglaTLit-PT (243K texts) | FPT corpus | Low — BanglishRev, MixSarc substitute |
| `csebuetnlp/banglishbert` + normaliser | Key model | Med — MuRIL substitutes |
| `google/muril-base-cased` | Key model | Low — Apache 2.0, stable |
| LLM API (generation + reference) | Generation | Med — open models substitute |
| Kaggle / Colab GPU | Compute | Low |

### 11.2 Internal

- 2–3 native Bangla annotators with Banglish fluency (M2 onward)
- 3–5 native writers for human-written Banglish items (M1–M3)
- Domain expert for adjudication

---

## 12. Open questions

| # | Question | Needed by | Resolution path |
|---|---|---|---|
| Q1 | What is the licence on the অলীকবচন Kaggle competition data? | M0 | Log in, read rules tab |
| Q2 | Does that competition include a code-mixed track, or is it Bengali-only? | M0 | Inspect the dataset |
| Q3 | What ratio of human-written vs. tool-transliterated Banglish is achievable given annotator availability? | M1 | Recruit first, then set target |
| Q4 | Does back-transliteration (Arm B) actually help, or does information loss hurt? | M5 | Empirical — that's what the arm sweep is for |
| Q5 | Should the "not sure" / abstention class be added? | M2 | MedHallu reports up to +38% relative F1 gain from it; test on the pilot |
| Q6 | Which FPT base model gives the largest gain on *this* task? | M6 | BanTH suggests mBERT and BanglishBERT; verify |

---

## 13. Definition of done

Phase 1 is complete when **all** of the following are true:

- [ ] Corpus of ≥ 4,000 annotated pairs exists with train/dev/test splits
- [ ] Inter-annotator κ ≥ 0.60 reported
- [ ] Shortcut probe scores < 0.60 macro-F1 on the final corpus
- [ ] All Must-priority models (M1–M5) trained and benchmarked
- [ ] Has-context macro-F1 ≥ 0.80 achieved
- [ ] No-context macro-F1 ≥ 0.60 achieved
- [ ] Results tables 1–5 produced
- [ ] 5-fold CV and 3-seed results reported with variance
- [ ] Test set was evaluated exactly once
- [ ] Experiment log complete, including failed runs
- [ ] Codebase reproducible from a clean environment
- [ ] Phase 1 technical report written, including a **limitations section**
- [ ] `data/SOURCES.md` complete with licences

---

## 14. Transition to Phase 2

Phase 1 hands Phase 2 the following, which is why the schema and locking discipline matter now:

| Asset | Phase 2 use |
|---|---|
| Corpus with `script_condition` + `cmi` fields | Basis for the CMI-controlled degradation curve |
| Arm A/B/C preprocessing infrastructure | Becomes the normalisation ablation experiment |
| Full model ladder with per-model results | Becomes the layered mechanism analysis |
| `error_span` fields | Span-level annotation seed |
| Human-written holdout | Natural vs. synthetic transfer experiment |
| Validated pipeline | Lets Phase 2 be about findings, not plumbing |

**Phase 2 headline claim (draft):** *Detection of hallucination in Bangla–English code-mixed text degrades measurably with code-mixing intensity; the degradation is attributable to specific representational levels; and detectors evaluated on synthetically code-mixed data overestimate their performance on naturally occurring Banglish.*

Phase 1's job is to make that claim checkable.

---

## Appendix — Terminology

| Term | Definition |
|---|---|
| **Banglish** | Romanised Bangla — Bangla written in Latin script, typically with English words interleaved |
| **Code-mixing** | Use of two or more languages within a single utterance |
| **CMI** | Code-Mixing Index — quantifies the degree of mixing in an utterance |
| **Intrinsic hallucination** | Output contradicting the provided context (a faithfulness failure) |
| **Extrinsic hallucination** | Output contradicting world knowledge with no context provided (a factuality failure) |
| **FPT** | Further pretraining — continued MLM training of a pretrained encoder on in-domain unlabeled text |
| **Shortcut / artifact** | A surface feature correlated with the label that lets a model succeed without solving the task |
| **IAA** | Inter-annotator agreement |
