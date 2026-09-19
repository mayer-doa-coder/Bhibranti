# CLAUDE.md — Phase 1 Cheat Sheet

**Phase 1 is Bengali only.** Bangla script, Bangla questions, Bangla answers. The job is to
detect when an answer is wrong (hallucinated).

**Banglish comes later, in Phase 2.** Do not start it. Full context in [AGENT.md](AGENT.md);
requirements in [docs/PRD.md](docs/PRD.md); recipes in
[docs/IMPLEMENTATION_GUIDE.md](docs/IMPLEMENTATION_GUIDE.md).

---

## Hard constraints

| | |
|---|---|
| **RAG is forbidden.** | No retrieval-augmented anything. No vector store, no nearest-neighbour lookup over the corpus, no retrieving passages or similar examples at inference. The model sees the question, the passage that is already in the record, and the answer. Nothing else. |
| **Allowed** | N-gram · Skip-gram · word2vec · word & text embeddings · RNN · LSTM/BiLSTM (+attention) · BERT-family encoders · anything else that is not RAG. Adding techniques is fine; the required ladder below may not be *replaced*. Edit distance, cosine similarity and an n-gram LM are fine **when they compare a record only with itself** (PRD §5.3a). |
| **Follows the NLP lab.** | The ladder covers the course's `NLP Lab/` syllabus (Labs 1–5): 26 lab topics used, 7 excluded with reasons — PRD §5.3b. Adapt lab code to Bengali; never copy it (§5.3c). |
| **Data collection is finished.** | The corpus is final. Do not collect, scrape, generate, or synthesise more data. Work with `data/corpus/bn_v1/`. |
| **QA only — no fill-in-the-blank.** | Cloze / শূন্যস্থান পূরণ items are excluded. They teach span-copying rather than answer checking. This removed `geography` entirely, which was 100% cloze. |

---

## Current status

**M0 done. M1 done. M2 done (kappa = 0.717). M3 done (test kappa = 0.865, merged, D8 met,
180 flagged pairs excluded). M4 under way — the ladder is built through M13: text tools,
features, input formats, scoring, the classical models, the M10 ablation, and the neural
models (M3 + M13). `src/train_transformer.py` is written and tested; the pretrained encoders
(M4/M5) need a GPU and run from `notebooks/02_bert_gpu.ipynb`.
Next up: fine-tune BanglaBERT on Colab.**

**Usable data (load it only through `src/splits.py`):** train 3,067 pairs / 6,134 records ·
dev 619 / 1,238 · test 614 / 1,228.

- [x] Repo scaffolded, `configs/schema.json` frozen, seeds fixed at **42, 1337, 2024**
- [x] 62,084 raw records → cleaned to 56,480 in `data/interim/bn_pool.jsonl`
- [x] **Corpus built → `data/corpus/bn_v1/corpus.jsonl`, 4,480 pairs / 8,960 records**
- [x] Splits built, grouped by pair → `data/splits/{train,dev,test}.jsonl` (3,129 / 673 / 678 pairs)
- [x] **Shortcut gate PASSES** — metadata probe 0.534 (needs < 0.60)
- [x] Easy/hard marked — the hard subset is where the string shortcut stops working
- [x] **M2 PASSED — Cohen's κ = 0.717** on 100 blind items, 2 annotators
      → [`data/annotated/agreement_test_v1/IAA_REPORT.md`](data/annotated/agreement_test_v1/IAA_REPORT.md)
- [x] Guidelines extended with Rules 11–15, taken from the actual disagreements
- [x] M3 sheets built → `data/annotated/round1/`
- [x] **All four annotation sets complete and validated**: test/tawhid (1,356), test/shejan
      (1,356), dev (1,346), train_spotcheck (1,252) — 5,310 rows, 0 structural errors, 0 tampered
      question/answer text, 0 cross-split leakage. Files use a `*_FINAL.csv` naming convention.
- [x] **Test split double-annotated — Cohen's κ = 0.865** (full 1,356 items, not a sample)
      → run `python src/score_test_agreement.py --a-dir .../test/tawhid --b-dir .../test/shejan`
- [x] **Merge tool rebuilt** — `src/merge_annotation.py` now reads the `*_FINAL.csv` sheets.
      Dry run verified; write path tested on a copy (labels/difficulty untouched, idempotent,
      gate still 0.534). It never changes `label` or `difficulty`.
- [x] **Adjudication done** — all 253 rows decided (141+2 typed, 89 dispute, 21 skip; 3 typo-pairs
      moved to skip on review). **Merge written 2026-09-17**; strict audit passes, gate 0.534.
      D8 MET: 1,867 of 1,977 in-scope wrong answers typed, 110 excluded by the adjudicator.
- [x] **D8 narrowed (PRD §5.1c):** types are required for test + dev + the 20% train spot-check
      (1,977 wrong answers), not the other 2,503 train records. Those stay `unlabeled` with
      `type_source = outside_train_sample`. Train *labels*: 4.5% confirmed noise (< 5% rule).
- [x] **PRD Q5 decided (§5.1d): human-flagged pairs are excluded from training AND scoring** —
      train 62, dev 54, test 64 (stored label confirmed wrong, or broken item). They stay in the
      files marked `excluded = true`; `src/splits.py` drops them. Gate on usable data: 0.514.
- [x] **Label disputes recorded, not applied:** 222 records where a human disagrees with the
      corpus label (116 unanimous) → `data/annotated/round1/label_disputes.csv`.
- [x] **M4 step 1 — `src/text_bn.py`** (Lab 1): Bengali cleaning, tokenizer, stop words, stemmer.
      Protected negation + number words (`configs/bn_protected_words.txt`); stemmer only cuts if
      the stem is a real train word. `--check` passes 9 rules on train+dev; 27 tests pass.
- [x] **M4 step 2 - `src/features.py`** (Labs 1-3): char n-gram LM (M11) + similarity features (M12).
      Settings tuned on TRAIN hard AUC: order 4, add-k k=0.1. **V6 measured**: fuzzy rule on dev
      hard = **0.591** (exact rule 0.487) - the hard-subset bar is 0.591, not 0.487.
      On hard, `lm_passage` is the best single feature (AUC 0.657); `exact_in_passage` collapses
      to 0.528. M11b answer-style classifier 0.529-0.546 ~= answer-only probe 0.542 (no
      generator fingerprint). 24 tests pass.
- [x] **M4 step 3 - `src/preprocess.py`** (guide §6, M9): the three input formats F1/F2/F3.
      Cuts the PASSAGE only, never the question or answer - verified on all 7,372 train+dev
      records. At the 256-token budget only 1.5% of has-context records need cutting, and
      measured evidence loss is 7 of 1,634 locatable pairs (0.4%); at 128 tokens it is 7.2%.
      Choosing the passage window by ANSWER overlap is forbidden (it would leak). 22 tests pass.
- [x] **M4 step 4 - `src/evaluate.py`** (guide §12): every model is scored here, so none can be
      judged on its own terms. macro-F1 + accuracy + per-class P/R + AUC, sliced by condition,
      difficulty, subject and error type, with the rule baselines recomputed on the same
      records. Three deliberate choices: the 95% interval resamples PAIRS (the two answers to
      one question are not independent); per-type slices report DETECTION RATE plus a
      pair-level score (a type slice holds only wrong answers, so macro-F1 there is
      degenerate); McNemar is written out because statsmodels is not a dependency. Adds an
      explicit `has-context + hard` row - the PRD G4 target. `--split test` refuses without
      `--final`. 28 tests pass.
- [x] **M4 step 5 - `src/train_classical.py` + `src/skipgram.py`** (Labs 2-3): the first models
      that learn. M1 BoW/TF-IDF x NB/LogReg/SVM, M11 char LM, M2 Skip-gram x LogReg/XGBoost,
      M12 similarity features. **gensim has no Python 3.14 build, so Skip-gram is written out
      from Lab 3** (negative sampling instead of full softmax); its neighbours are sensible
      (1971 -> মার্চ, জিয়াউর; সরকার -> মুজিবনগর, প্রবাসী) and it covers 94% of dev words.
      Lab-vs-library Naive Bayes check: **100% agreement**.
      **Best on the target (has-context + hard): tfidf_logreg and tfidf_svm tie at 0.610**
      (McNemar p = 1.000 - 14 wins each on the 28 they disagree about). M12 tops the overall
      table at 0.730 but sinks to 0.490 on hard: it learned the string-match shortcut.
- [x] **Repo-wide bug sweep (2026-09-19)** - 5 fixed: stray `<SEP>` markers polluting every M1
      feature column (M1 re-measured, moved <= 0.014); word-vector filenames disagreeing between
      writer and readers (now one `vectors_path(seed, variant)`); run-id generator could raise
      StopIteration; `xgboost`/`rapidfuzz` missing from Tier 1 of requirements.txt so a fresh
      clone would fail; an unused import. Verified clean: no dev leakage, test split guarded at
      every entry point, reruns reproduce exactly, all 12 documented settings match the code.
      **Second pass, 5 more:** `results/experiment_log.csv` was opened for reading before it was
      known to exist, so a deleted `results/` folder would throw away a finished run at the last
      step (now `evaluate.ensure_log_file()`, and `LOG_FILE` is defined once instead of three
      times); `load_or_train_vectors` still spelled the vector filename by hand instead of calling
      `vectors_path()` - the same drift that caused bug 2; long runs buffered all output, so a
      working ablation looked hung (progress lines now flush); `AGENT.md` §10.5 required
      re-running the audit after changes to `src/generate.py`, a file that never existed;
      `features.py` imported `Path` and never used it. Checked and found correct: every path in
      the docs resolves, every documented CLI flag exists, every score quoted in the guide and
      walkthrough matches `results/experiment_log.csv`, no duplicate run ids, and all module
      paths are `__file__`-anchored so the scripts work from any folder. 148 tests pass.
- [x] **M4 step 6 - M10 preprocessing ablation** (`--ablation`, Table 6): V0-V4 + V2-demo x
      tfidf_logreg and skipgram_mean_xgb, dev, seed 42.
      **The textbook cleanup is worth nothing measurable here** - tfidf_logreg moves 0.542-0.551
      overall across V0-V4 and holds 0.610 on hard for three of five. V1 stays the default.
      **V2-demo is the finding: dropping negation costs 0.184 on `contradiction`** (0.712 ->
      0.528, -26% relative) **while its hard score stays at 0.609 vs the default's 0.610** - the
      damage is invisible in the headline number and only appears in the error-type slice. Best
      argument in the project for §12.1. Stop-word removal actively hurts Skip-gram (0.501 on
      hard, worst cell; its best variant is V0, no cleaning at all). V4 does buy a smaller model:
      201,504 features against V0's 259,454 for the same score.
- [x] **Demo interface** (`src/app.py`, `src/serving.py`, `web/`): type a question + answer
      (+ optional passage), all 12 models judge it, with the M12 evidence and the two rule
      baselines shown beside them. `/models` is the scoreboard, read from
      `results/experiment_log.csv` and averaged over seeds so it can never disagree with
      Table 5. Two-column layout, fits one laptop screen (576px) with no scrolling.
      `train_and_predict` was split into `fit_model()` + `FittedModel.predict()` so serving
      and training are ONE code path; verified behaviour-preserving (tfidf_logreg still
      0.549 / 0.610 / 215,601 features).
      **No Prometheus/Grafana** - removed 2026-09-19 at the user's request. Do not add
      metrics endpoints, `src/telemetry.py`, or an `ops/` stack back.
- [x] **M4 step 7 - `src/train_neural.py`** (Labs 4-5): M3 rnn/birnn/bilstm/bilstm_attn and
      M13 transformer_scratch, CPU, 3 seeds each. **rnn is the best at 0.560 / 0.605 hard** -
      the SIMPLEST architecture won; bidirectionality, stacking and attention all cost a
      little. None beats the best classical model (0.610). **transformer_scratch is the
      weakest AND least stable: 0.465 ±0.072, collapsed outright on seed 2024 (0.364)** -
      flagged by `collapse_warning()`, not hidden. Three fixes to the lab code, each
      demonstrated by `--check` on real records: packed sequences (the lab reads its final
      state off padding), masked mean pooling, and an embedding init that stops token vectors
      drowning the position signal 11:1. 67 tests, mutation-tested.
- [x] **M4 step 8 - `src/train_transformer.py`** (M4/M5): BanglaBERT, MuRIL, XLM-R, mBERT.
      PRD recipe by default, `--two-stage` (freeze head, then unfreeze top layers) as the
      small-data fallback from the reference notebooks. **Refuses to run BanglaBERT without
      csebuetnlp/normalizer** rather than silently scoring lower - both reference notebooks
      miss this. No back-translation and no LLM blended in (both tested for). 24 tests.
- [x] **Notebooks** - `00_full_walkthrough.ipynb` (the whole pipeline end to end with charts,
      runs in ~9 min, verified to execute with 0 errors), `01_neural_gpu.ipynb`,
      `02_bert_gpu.ipynb` (Colab + Drive). They import `src/`, never re-implement it.
- [ ] Next: run `02_bert_gpu.ipynb` on Colab (M4/M5), then M6 FPT, M7 ensemble, M8 LLM reference

**All 13 remaining subjects are in.** Nothing is excluded for being hard or for needing outside
knowledge — law, science, BCS and literature are all included on equal footing (966 pairs,
21.6% of the corpus). Difficulty is *measured* (see `difficulty`), not filtered away.

**`geography` is gone**, and that is not a difficulty judgement: every geography item was
fill-in-the-blank, and this project is QA only.

**Three numbers your has-context score must beat.** Report them, always. Measured on the usable
data (excluded pairs removed), because that is what models train and are scored on:

| Baseline | Score | What it means |
|---|---|---|
| String matcher, all has-context | **0.823** | "Does the answer appear in the passage?" — no learning at all |
| String matcher, easy subset | 0.987 | where the shortcut fully works |
| String matcher, **hard subset** | **0.454** | on hard pairs the shortcut is useless |
| **Fuzzy** matcher, **hard subset** (V6) | **0.591** on dev hard | allows ~40% of characters to differ; exact rule gets 0.487 on the same records. **This is the real hard-subset bar** |

(Before exclusion they were 0.812 / 0.980 / 0.456. For a dev or test score, recompute the baseline
on that same split.) A has-context macro-F1 of 0.83 is **not** a result — it barely beats a string
matcher. The real
target is the **hard subset**, where 0.80 actually means something.

---

## Setup

```bash
python -m venv .venv && source .venv/Scripts/activate     # Windows/Git Bash
pip install -r requirements.txt        # Tier 1 only; Tier 2 is commented out until M4
```

## Commands

```bash
# Data pipeline — THESE FOUR EXIST AND WORK TODAY (deterministic, seed 42)
python src/build_bn_pool.py         # raw -> data/interim/bn_pool.jsonl
python src/build_corpus.py          # interim -> data/corpus/bn_v1/ + data/splits/
python src/build_agreement_test.py  # corpus -> 100 blind items for the M2 check
python src/audit.py --data data/splits              # all probes + structural checks

# THE GATE — metadata probe must be < 0.60, exits non-zero if not
python src/audit.py --data data/splits --probe metadata

# M3 — build the annotation sheets, then check filled ones
python src/build_annotation_sheets.py        # refuses to clobber labelled work
python src/validate_annotation.py --dir data/annotated/round1/test/tawhid
python src/score_test_agreement.py --a-dir data/annotated/round1/test/tawhid \
  --b-dir data/annotated/round1/test/shejan --a-name Tawhid --b-name Shejan   # kappa 0.865
python src/merge_annotation.py --dry-run     # sheets -> corpus report; drop --dry-run to write
                                             # also sets `excluded` on human-flagged pairs
                                             # rerun it after ANY build_corpus.py rebuild

# M2 gate — PASSED at kappa 0.717. Rerun any time:
python src/score_agreement.py \n  --a data/annotated/agreement_test_v1/items_for_annotation_tawhid.csv \n  --b data/annotated/agreement_test_v1/items_for_annotation_shejan.csv

# M4 step 1 - Bengali text tools (Lab 1). Every non-pretrained model uses preprocess() from here
python src/text_bn.py "কলেজের ছাত্ররা বইটি পড়েনি।"   # show every cleaning step and variant
python src/text_bn.py --check                         # 9 safety rules on train+dev (exit 1 on failure)
python -m pytest tests/ -v                            # unit tests, one per rule

# M4 step 2 - hand-made features + the V6 baseline (Labs 1-3)
python src/features.py --demo        # features side by side for a correct vs wrong answer
python src/features.py --check       # sanity rules + how much signal each feature carries
python src/features.py --baseline    # tune + score the V6 fuzzy rule (add --log to record it)
python src/features.py --tune-lm     # redo the language-model order/smoothing table

# M4 step 3 - the three input formats (guide §6). Passage is cut; question/answer never are
python src/preprocess.py --demo                       # one record in all three formats
python src/preprocess.py --check                      # the guarantees, on train+dev
python src/preprocess.py --format F3 --split dev --out data/processed/

# M4 step 4 - scoring. Every model reports through this, so the numbers stay comparable
python src/evaluate.py --demo                   # what the numbers mean, on made-up models
python src/evaluate.py --baselines --table      # score the no-learning rules -> results/tables/
python src/evaluate.py --baselines --split test --final   # ONLY at M6, once, ever

# M4 step 5 - the first models that learn (CPU only, no GPU)
python src/skipgram.py --train                   # ~3 min; writes data/processed/skipgram_s42.npz
python src/skipgram.py --check                   # are the word neighbours sensible?
python src/train_classical.py --model tfidf_logreg          # one model, full report
python src/train_classical.py --all --seeds --log           # every model, 3 seeds, recorded
python src/train_classical.py --model tfidf_logreg --variant V2   # M10 ablation
python src/train_classical.py --lab-check        # our Naive Bayes vs the library's

# THE DEMO - type an answer, see what all 12 models say (guide sec 15, walkthrough 18.9)
python src/serving.py --build                    # once, ~6 min: fit all 12 and save them
python src/serving.py --check                    # load them back, predict one record
python src/app.py                                # then open http://127.0.0.1:8000

# Not yet written (M4-M6) - see guide §1.3 for the planned modules
python src/train_classical.py    --model tfidf_nb --seed 42          # M1, M2, M11, M12 (Labs 1-3)
python src/train_classical.py    --model tfidf_nb --variant V2       # M10 preprocessing ablation
# M4 step 7 - the neural models (CPU only, ~1 min/epoch for the heaviest)
python src/train_neural.py --check                      # prove the 3 fixes to the lab code
python src/train_neural.py --demo                       # one record through every model
python src/train_neural.py --model bilstm_attn          # M3 (Lab 4), one model, full report
python src/train_neural.py --model transformer_scratch  # M13 (Lab 5)
python src/train_neural.py --all --seeds --log          # all 5, 3 seeds, recorded
python src/train_neural.py --model bilstm_attn --attention   # where did it look?
python src/train_transformer.py  --model csebuetnlp/banglabert --seed 42
python src/evaluate.py --checkpoint out/best --split dev --shuffle-order # M14 word-order test
python src/evaluate.py --checkpoint out/best --split dev      # any number of times
python src/evaluate.py --checkpoint out/best --split test     # EXACTLY ONCE, at M6
```

> Every script takes `--seed` and appends a row to `results/experiment_log.csv`.

## Repo layout

| Path | Purpose |
|---|---|
| [docs/PRD.md](docs/PRD.md) | Requirements, gates, scope — **the authority** |
| [docs/IMPLEMENTATION_GUIDE.md](docs/IMPLEMENTATION_GUIDE.md) | Recipes, hyperparameters, code |
| [docs/ANNOTATION_GUIDELINES.md](docs/ANNOTATION_GUIDELINES.md) | Labelling rules — read before annotating |
| `configs/schema.json` | Frozen record schema |
| `src/build_bn_pool.py` | ✅ Cleans + dedups the raw pool |
| `src/build_corpus.py` | ✅ **Pairs, filters, splits.** Read its docstring first |
| `src/build_agreement_test.py` | ✅ Builds the 100-item blind test |
| `src/score_agreement.py` | ✅ Cohen's kappa — the M2 gate |
| `src/build_annotation_sheets.py` | ✅ Builds the M3 sheets. Read its docstring — 6 documented guards |
| `src/validate_annotation.py` | ✅ Checks filled sheets for the mistakes that actually happen |
| `src/score_test_agreement.py` | ✅ Cohen's kappa on the full test split |
| `src/merge_annotation.py` | ✅ Human sheets → `hallucination_type` + `excluded` in corpus + splits. Never touches `label` |
| `src/splits.py` | ✅ `load_split("train")` — **the only way to load data for training or scoring**; drops excluded pairs |
| [docs/PROJECT_WALKTHROUGH.md](docs/PROJECT_WALKTHROUGH.md) | Beginner-friendly explanation of every step so far — **keep it updated as steps finish** |
| `src/audit.py` | ✅ Shortcut probes + structural checks |
| `src/text_bn.py` | ✅ Bengali cleaning, tokenizer, stop words, stemmer (Lab 1). Use `preprocess(text, variant)` |
| `configs/bn_stopwords.txt` · `bn_protected_words.txt` · `bn_suffixes.txt` | ✅ Word lists for `text_bn.py`. Edit the protected list, never the published stop list |
| `tests/test_text_bn.py` | ✅ One test per text rule — run after any change to `text_bn.py` or the word lists |
| `src/features.py` | ✅ M11 char n-gram LM, M12 edit distance features, V6 fuzzy baseline (Labs 1–3). `extract_features(record)` |
| `tests/test_features.py` | ✅ One test per feature rule — run after any change to `features.py` |
| `src/preprocess.py` | ✅ Input formats F1/F2/F3. `as_tokens()` for classical models, `format_record()` for encoders. Cuts the passage only |
| `tests/test_preprocess.py` | ✅ One test per formatting rule |
| `src/train_classical.py` | ✅ M1 BoW/TF-IDF + NB/LR/SVM, M2 Skip-gram + LR/XGB, M11, M12 (Labs 2–3). `--all` compares them |
| `src/skipgram.py` | ✅ Lab 3's Skip-gram, written out. Vectors are per-seed: `data/processed/skipgram_s42.npz` |
| `tests/test_skipgram.py` · `tests/test_train_classical.py` | ✅ 35 tests for the word vectors and the training harness |
| `src/serving.py` | ✅ Fits every model once and caches it in `data/processed/serving/` (gitignored, 83 MB). Uses `train_classical.fit_model()` — **never rebuild the pipeline here or the demo drifts from Table 5** |
| `src/app.py` | ✅ The demo: `/` judge an answer, `/models` the scoreboard, `/health` |
| `web/templates/` · `web/static/` | ✅ The demo's pages and its one stylesheet |
| `tests/test_serving.py` | ✅ 29 tests. The ones needing real models skip themselves if `--build` has not been run |
| `src/train_neural.py` | ✅ M3 RNN/BiRNN/BiLSTM(+attn), M13 Transformer from scratch (Labs 4–5). CPU, PyTorch. `--check` proves the 3 fixes to the lab code |
| `tests/test_train_neural.py` | ✅ 65 tests. Mutation-tested: breaking any of the 3 fixes makes a named test fail |
| `src/train_transformer.py` | ✅ Pretrained encoders (M4/M5): BanglaBERT, MuRIL, XLM-R, mBERT. `--two-stage` for the small-data recipe. **Refuses BanglaBERT without csebuetnlp normalizer** |
| `tests/test_train_transformer.py` | ✅ 24 tests. No download, no GPU — the guards that would otherwise cost accuracy silently |
| `src/colab_setup.py` | ✅ `prepare()` — mounts Drive, adds `src/` to the path, prints a data hash to confirm Drive matches your laptop |
| `notebooks/01_neural_gpu.ipynb` · `02_bert_gpu.ipynb` | ✅ Colab GPU drivers. Thin: they import `src/`, never re-implement it |
| `src/further_pretrain.py` | ⬜ MLM further pretraining (mBERT / XLM-R only) |
| `src/evaluate.py` | ✅ Metrics, breakdowns, baselines, M14 shuffling, McNemar, pair-level bootstrap. `report(records, predictions)` |
| `tests/test_evaluate.py` | ✅ One test per scoring rule, checked against hand-worked values |
| `NLP Lab/` | Course lab guides + notebooks — the syllabus the ladder follows. Read-only |
| `data/DATASET_AUDIT.md` | **Read first.** Audit of the source pool |
| `data/corpus/bn_v1/corpus.jsonl` | ✅ The Phase 1 corpus, 4,480 pairs |
| `data/splits/` | ✅ `train/dev/test.jsonl` — the real splits |
| `data/annotated/round1/` | ✅ The M3 annotation sheets. Never open `_mapping_DO_NOT_SHARE.csv` |
| `data/interim/bn_pool.jsonl` | 56,480 cleaned records (the pool the corpus is drawn from) |
| `data/raw/bn_qa_pool/` | 14 source `.jsonl`, verbatim. Has duplicates — don't load directly |
| `data/SOURCES.md` | Licence register — 2 open questions |
| `results/experiment_log.csv` | Single source of truth for every run |

## Key facts

- `label`: **1 = correct/faithful**, **0 = hallucinated**. Positive class is *faithful* — the
  reverse of most papers. `probs[:, 1]` = P(correct).
- Corpus: 60% has-context / 40% no-context · 50/50 class balance · one pair = one correct + one
  hallucinated answer to the same question
- **Pairs never split across train/test.** Splitting by record would put the correct answer in
  train and the wrong one in test, so a model could score by memorising the question.
- `difficulty`: **hard** = the string shortcut does not separate the two answers on this pair.
  For no-context, hard = the wrong answer is a near-miss.
- **No fill-in-the-blank items.** QA only. They were 0.929 for a plain string matcher, which
  inflated every has-context score.
- Seeds: **42, 1337, 2024** — always all three, report mean ± std
- Fine-tune defaults: `lr=2e-5`, AdamW, `max_length=256`, `epochs=5`, `batch=32`, `warmup=0.1`,
  `weight_decay=0.01`, `fp16=True`, early stop on val loss (patience 2)
- Targets: has-context **≥ 0.80 on the hard subset**, no-context **≥ 0.60**, metadata probe
  **< 0.60**, κ **≥ 0.60**
- Taxonomy: intrinsic `entity|numeric|relational|contradiction`; extrinsic `fabricated|overclaim`;
  correct `none`. Types exist for test, dev and the train spot-check only (PRD §5.1c) — per-type
  results are reported on **dev/test only**
- **`hallucination_type`, `type_source`, `annotator_1/2`, `adjudicated` reveal the label — never a
  model input**
- **Model ladder (PRD §5.3), in lab order:** M1 BoW/TF-IDF + Naive Bayes/LR/SVM → M11 char n-gram
  LM → M2 Skip-gram (mean / TF-IDF-weighted) + LR/XGBoost → M12 edit-distance + cosine features →
  M3 vanilla RNN, BiRNN, BiLSTM, BiLSTM + dot-product attention → M13 Transformer from scratch →
  M4/M5 pretrained encoders → M6 FPT → M7 ensemble → M8 LLM reference. Plus M9 input formats,
  M10 preprocessing ablations, M14 word-order diagnostic
- **Baselines to report beside has-context scores:** exact string matcher (0.823 / 0.454 hard) and
  the fuzzy string matcher (V6, measured at M4)
- Recurrent / from-scratch defaults: Adam `lr=1e-3` (RNNs) / `5e-4` + warm-up (Transformer),
  `batch=32`, early stop on dev loss (patience 3), single logit + `BCEWithLogitsLoss`,
  sigmoid = P(correct)

---

## DO

- **Report has-context and no-context separately**, and **easy and hard separately**. A single
  averaged number hides everything that matters here.
- **Load data with `from splits import load_split`.** It drops the 180 excluded pairs (PRD §5.1d).
  Reading `data/splits/*.jsonl` directly would train and score on labels humans confirmed wrong.
- **Quote the string baselines** (0.823 all / 0.454 hard on usable data, or recomputed on the
  split being scored) next to every has-context score you report.
- **Log every run** to `results/experiment_log.csv` — including crashes and failures.
- **Set and log a seed** in anything that touches randomness.
- **Run `normalize()`** from `csebuetnlp/normalizer` before BanglaBERT — it was pretrained with
  it, and skipping it costs points.
- **Break results down by subject too.** Law, science and BCS items are harder to verify; if the
  model collapses on those, that is a finding worth reporting, not something to hide.
- **Use McNemar + bootstrap CI** before claiming one model beats another.
- **Keep digits and negation words (না, নয়, নি, নেই, নাই) in every preprocessing variant** except the
  one labelled V2-demo. They carry the `numeric` and `contradiction` errors.
- **Apply Unicode NFC** to text and to any stop-word / suffix list before matching.
- **Report answer-only models** (M11b class-conditional LM) next to the V3 answer-only probe.
- **Say which lab each result comes from** (Tables 1 and Appendix B of the guide) — the ladder is
  also a course deliverable.
- **Investigate any macro-F1 > 0.95** as leakage.

## DON'T

- **DON'T use RAG.** No retrieval at inference, no vector database, no nearest-neighbour lookup
  over training examples. The passage in the record is the only context the model gets. For the
  LLM reference (M8): zero-shot, or **one fixed** few-shot set for every item — never examples
  picked per item by similarity, and no search/browsing tools (PRD §5.3a).
- **DON'T collect or generate more data.** The corpus is final.
- **DON'T add fill-in-the-blank items back.** QA only — cloze is a different task.
- **DON'T touch `data/splits/test.jsonl`** except for the single sanctioned M6 evaluation.
- **DON'T weaken the shortcut probe** to make it pass — not the threshold, not the features, not
  the split. Fix the data and rebuild.
- **DON'T report a raw has-context number on its own.** A string matcher gets 0.823. Without the
  hard-subset number next to it, the score is meaningless.
- **DON'T use `AutoModelForMaskedLM` on BanglaBERT** — it is an ELECTRA discriminator. Further
  pretraining targets mBERT and XLM-R only.
- **DON'T swap out a required model.** The ladder (N-gram → Skip-gram → BiRNN/BiLSTM →
  BERT-family, extended to cover the NLP lab in M10–M14) is fixed by the course requirement.
  Adding is fine; replacing is not.
- **DON'T copy lab code onto Bengali.** `re.sub(r'[^A-Za-z\s]', '', text)` deletes all Bengali and
  all digits; NLTK `word_tokenize`, English `stopwords`, `PorterStemmer`, `WordNetLemmatizer` are
  English-only. Use the Bengali recipes in guide §6.
- **DON'T compare a record with other records** for features (edit distance, cosine, LM). Only its
  own question and passage. Anything else is retrieval.
- **DON'T add the excluded lab topics** — lemmatization, spelling correction, text generation,
  temperature sampling, POS tagging, Seq2Seq translation (PRD §5.3b says why).
- **DON'T use pretrained fastText vectors** — decided 2026-09-17; Skip-gram is trained on the
  train split.
- **DON'T rename, remove, or repurpose a schema field.** `cmi`, `script_condition`, and
  `error_span` are Phase 2 payload: populate them, never evaluate on them.
- **DON'T hand-curate a split** or re-roll a seed until the numbers improve.
- **DON'T "fix" expected results.** Classical models at 0.50–0.65, and no-context well below
  has-context, are correct outcomes, not bugs.
- **DON'T start Phase 2 work** — Banglish/transliteration, CMI degradation, span-level
  evaluation, tokenizer fertility, paper writing. Flag it and stop.
- **DON'T commit API keys**, PII, or non-redistributable source data. Wikipedia content is
  CC BY-SA 4.0 — **share-alike**. Never mix it into one file with CC BY-**NC**-SA sources.
- **DON'T load `data/raw/bn_qa_pool/` directly** — 8.9% duplicates, no schema. Use `data/interim/`.
- **DON'T flip labels anywhere.** `1 = correct` end to end, source files included.

---

## When stuck

| Symptom | Check |
|---|---|
| Metadata probe > 0.60 | Inspect `clf.coef_` → find the leaking feature → fix the data → rebuild |
| has-context macro-F1 ≈ 0.81 | That is the string-matcher baseline. Look at the hard-subset score instead |
| macro-F1 > 0.95 | Leakage. Check for duplicate questions across splits; run the answer-only probe |
| Answer-only probe ≈ full model | Model isn't checking grounding — the answer text gives the label away |
| Hard subset far below easy subset | Expected and correct. That gap is the real finding |
| Model collapses on law / science / bcs | Expected — those need outside knowledge a closed-book model does not have. Report per-subject |
| κ < 0.60 | Guidelines, not annotators. Write explicit rules for the cases they disagreed on |
| Many `unsure` labels from one subject | A finding about the corpus. Report it; don't force a label |
| Model spread within 1–2 points | Expected. Use 3 seeds + McNemar; report ties honestly |
| CUDA OOM | `batch_size=16` + gradient accumulation. Don't change `max_length` mid-experiment |
| Score plateaued | ROI order: input format sweep → FPT → hard-negative balance → ensemble → threshold |
| Per-class metrics look swapped | `1` is the **correct** class here, not the hallucinated one |
| Stop words / stemming lower `contradiction` F1 | Expected if negation was stripped — check the list lost না/নয়/নি; that is exactly why V1 is the default |
| BoW model changes under M14 shuffling | Bug — a bag-of-words model cannot see order. Check the shuffle is done on the tokens *before* the same fitted vectorizer runs, and that no word was dropped or duplicated |
| Transformer from scratch far below BanglaBERT | Expected — that gap is what pretraining is worth. Don't tune it away |
| Stop-word list matches nothing | Unicode form mismatch (য় as U+09DF vs য + ়). Apply NFC to the list |
| Beats 0.454 on hard but not the fuzzy matcher | The model learned fuzzy string matching, not grounding (V6) |
