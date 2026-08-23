# `data/annotated/` — post-human-annotation corpus

## `agreement_test_v1/` — ready now: the 100-item guideline check

See **[`agreement_test_v1/README.md`](agreement_test_v1/README.md)**. This is the M2 step:
two people label the same 100 blind items using `docs/ANNOTATION_GUIDELINES.md`, and the
agreement between them (Cohen's kappa) tells us whether the guidelines actually work.

**Do this before anything else below.**

---

**Below: what this folder becomes once M2 passes and the full corpus is labeled.**

Right now, beyond the file above, nothing in this project has been human-annotated.

PRD **D4** and **D5** are Must-priority and currently unmet: there is no annotation, so Cohen's
κ cannot be computed and no test set can be called verified.

## Prerequisites (milestone M2)

- [ ] `docs/ANNOTATION_GUIDELINES.md` written **before** annotation begins (PRD R4). BanTH's
      Appendix B is the template to imitate — per-category definitions with worked bilingual
      examples.
- [ ] 2 annotators minimum (3 preferred), native Bangla speakers fluent in Banglish. Recruit 3,
      require 2 (RK3).
- [ ] 100-item pilot annotated → compute κ → revise guidelines → *then* annotate the rest.
- [ ] A domain expert designated for adjudication.

## Annotation levels (guide §5.3)

| Split | Level |
|---|---|
| Test (500) | **100% human, double-annotated, adjudicated** |
| Dev (500) | 100% human, single annotator + spot check |
| Train (3,000) | Generator label + human verification on a 20% sample |

If the 20% train sample shows > 5% label noise, verify more.

## Agreement

Report **Cohen's κ** (2 annotators) or **Fleiss' κ** (3+), plus Krippendorff's α as a robustness
check. Floor is **κ ≥ 0.60**; target 0.70. BanTH reported 0.71 inter-annotator on binary labels.

Below 0.40 means the guidelines are broken, not the annotators. The usual cause is that
"hallucination" is under-defined for edge cases — partially correct answers, answers that are
true but not entailed by the context, answers that hedge. Write an explicit rule for each.

Annotators set `label` (**`1` = correct, `0` = hallucinated**), and for **`label=0`** the
`hallucination_type` from the 6-type taxonomy. The source pool provides **neither** type nor difficulty, so both are annotated from
scratch here.
