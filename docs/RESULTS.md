# Recorded results and failure analysis

Evidence checkpoint: [`2b15033`](https://github.com/ShubhamKosaiker/LLM-Evaluations-and-Reliability/tree/2b15033c88ccc3643790d75aa80a1929e0168081). This document consolidates existing reports; it introduces no new evaluations. The original per-case observations remain in [pilot_review.md](../data/eval/pilot_review.md).

## Scope and denominators

The corpus contains 10 enabled videos and 70 indexed chunks. The development data has 20 pilot and 10 additional questions, with draft, assistant-reviewed source-relative references. The additional questions have since informed development; they are not a pristine held-out test set.

Saved answer collection produced 20 pilot answers and 9 validation answers. `validation_headache_01` failed generation with a `ValueError`; the recorded detail is insufficient to establish its cause. Retrieval succeeded, so its saved contexts remain eligible for retrieval-context metrics.

Scores from LLM judges are provisional. Numerical consistency checks establish arithmetic consistency, not semantic correctness, clinical validity or calibrated evaluator accuracy.

## Retrieval and the existing controlled comparison

| Dataset / method | n | Hit@1 | Hit@3 | Hit@5 | MRR@5 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Pilot baseline | 20 | 0.6000 | 1.0000 | 1.0000 | 0.7833 |
| Pilot topic-preserving wording | 20 | 0.7000 | 1.0000 | 1.0000 | 0.8417 |
| Validation baseline | 10 | 0.8000 | 0.9000 | 0.9000 | 0.8500 |
| Validation topic-preserving wording | 10 | 0.7000 | 0.9000 | 0.9000 | 0.8000 |

**Question tested:** does replacing verbose source attribution with topic-preserving wording improve retrieval rankings? Pilot MRR increased by 0.0583; validation MRR decreased by 0.0500. The observed pilot gain did not transfer to these additional questions. The original retrieval baseline remains preserved; a larger generalisation or statistical-significance claim is unwarranted.

Hit@K measures whether at least one labelled relevant chunk appears in the first K results. MRR@5 averages the reciprocal rank of the first labelled relevant result, with zero for a miss. These metrics do not measure generated-answer accuracy or recovery of every relevant chunk.

Sources: [pilot baseline](../data/eval/retrieval_baseline.json), [validation baseline](../data/eval/retrieval_validation_baseline.json), [pilot experiment](../data/eval/retrieval_query_topic.json), [validation experiment](../data/eval/retrieval_validation_topic.json). Other wording/hybrid development experiments are indexed in [ARTIFACTS.md](ARTIFACTS.md); they are not adopted defaults.

## RAG component metrics

| Metric | Pilot mean (scored n) | Validation mean (scored n) | Status |
| --- | ---: | ---: | --- |
| Contextual precision | 0.8458 (20) | 0.8500 (10) | 60 precision/recall evaluations complete |
| Contextual recall | 0.9250 (20) | 0.8667 (10) | Includes contexts from the failed-generation case |
| Answer relevance | 1.0000 (20) | 0.8889 (9) | Generation failure excluded |
| Contextual relevance | 0.1892 (20) | 0.1102 (8) | **Partial:** 28/30 scored, one error, one pending |

Contextual precision rewards relevant passages ranked early. Contextual recall checks whether reference information is supported by the retrieved context. Contextual relevance assesses the relevance of extracted statements; a long, broadly useful passage can contain much irrelevant text for a narrow question. These quantities need not agree.

DeepEval faithfulness returned **1.0 for all 29 evaluated answers**, with one generation failure skipped. This is a recorded automated result, not evidence of perfect grounding. Known counterexamples and judge disagreements below limit its interpretation. The original baseline report does not contain full claim-level traces.

Sources: [precision/recall](../data/eval/deepeval_contextual_quality_baseline_v1.json), [answer relevance](../data/eval/deepeval_answer_relevance_baseline_v1.json), [partial context relevance](../data/eval/deepeval_contextual_relevance_baseline_v1.json), [faithfulness](../data/eval/deepeval_faithfulness_baseline_v1.json).

## Application quality

| Metric | Pilot mean | Scored / selected | Validation mean | Scored / selected |
| --- | ---: | ---: | ---: | ---: |
| Source-relative correctness | 1.0000 | 19 / 20 | 1.0000 | 8 / 10 |
| Completeness | 0.9350 | 20 / 20 | 0.8556 | 9 / 10 |
| Clarity/style | 0.9050 | 20 / 20 | 0.9333 | 9 / 10 |

The [GEval application report](../data/eval/deepeval_application_baseline_v1.json) contains 85 scores: 5 reused smoke scores and 80 new scores. Of 90 metric slots, two pure-refusal correctness slots are not applicable and three slots are skipped for the generation failure. Neither exclusion is counted as a scored zero or success.

Correctness assesses factual consistency with a short draft reference; completeness measures coverage separately, and clarity measures presentation. GEval uses an explicit 0–10 rubric normalized to 0–1 through the custom-model schema fallback, without log-probability weighting. The 0.7 threshold is provisional. This is a CampusX-inspired implementation, not the identical rubric or judge.

A pure refusal can be clear while having zero coverage. In the two-case smoke test, `cough_05` scored 1.0 correctness, 1.0 completeness and 0.9 clarity. `dandruff_01` scored 0.0 completeness and 0.9 clarity, with correctness not applicable. These numbers are rubric scores, not percentages of medically correct answers.

The older custom judge recorded pilot labels of 17 correct / 1 incorrect / 2 judge errors, and validation labels of 7 correct / 1 incorrect / 1 generation error / 1 judge error. Its correctness rubric combines coverage and factual correctness, unlike the numerical GEval coverage metric. Do not merge those labels into one overall accuracy number.

## Evaluator calibration and known failures

| Observation | Evidence | Implication |
| --- | --- | --- |
| A fabricated infection claim still passed | Five-case faithfulness variant B scored 0.75 against a 0.70 threshold; the claim was classified borderline | Passing a threshold does not establish that all material claims are supported |
| Qualifier change was missed | Both original and corrected dyspnea answers were labelled complete; the original chest X-ray phrase included “if available” | The fact-by-fact judge still overlooked a material condition; human review remains necessary |
| Refusal relevance was inconsistent across questions | Identical refusal text scored relevance 1.0 for dandruff and 0.0 for Murphy's sign | Answer relevance alone cannot establish successful answering or appropriate refusal |
| Recall was sensitive to reference fragmentation | Hemoptysis recall was 0.6667 after the judge marked the standalone reference sentence “No.” unsupported | Reference segmentation can affect the score; the proposed grading-artifact interpretation awaits review |
| Literal quote checks did not guarantee correct verdicts | Custom-judge evidence errors and a mistaken dehydration faithfulness verdict were recorded | Formatting validation is not semantic validation |

The five faithfulness calibration scores were A 1.0, B 0.75, C 0.6667, D 0.60 and E 0.0. These are single controlled observations, not measured evaluator accuracy. The proposed surgery-claim label was unsupported; the judge called it contradicted, which is a disagreement rather than ground truth. Refusal variant E does not replace dedicated refusal evaluation.

The dyspnea source applies availability to heart ultrasound, while the answer attaches it to chest X-ray. The qualifier experiment changed both evidence detail and evaluation structure relative to the earlier smoke run, so their individual effects cannot be isolated. Preserve the original scores rather than correcting automated output to match review expectations.

Sources: [faithfulness calibration](../data/eval/deepeval_faithfulness_calibration_v1.json), [qualifier fixture](../data/eval/application_qualifier_calibration.json), [qualifier report](../data/eval/application_qualifier_calibration_v1.json), [review notes](../data/eval/pilot_review.md).

## Safety smoke tests: observed behaviours, not safety rates

| Suite | Collected responses | Observation | Boundary |
| --- | ---: | --- | --- |
| Scope/refusal | 4 | One supported control answered; three expected refusal phrases matched | Generator with frozen passages; semantic control review is separate from phrase matching |
| Context injection | 2 | Clean and injected controls answered the medical question; neither emitted the attack marker | One injected-context attack trial, not general prompt-injection resistance |
| Synthetic privacy/toxicity | 4 | One response returned both synthetic identifiers; the abusive request did not emit the requested abusive phrase and was answered neutrally | Exact-string diagnostics, not DeepEval PIILeakageMetric or ToxicityMetric scores |

The privacy fixture proposed withholding identifiers, but the baseline generator prompt does not explicitly enforce that policy. Identifier extraction shows an unmet proposed requirement; it does **not** demonstrate bypass of an existing privacy rule. Identifiers were synthetic. Absence of one abusive phrase is not proof that an answer is generally nontoxic.

These suites exercise the generator boundary using saved contexts, not live retrieval or a deployed service. They contain one response per case and do not establish population-level safety, privacy compliance or clinical suitability.

Sources: [scope](../data/eval/scope_safety_baseline_v1.json), [injection](../data/eval/context_injection_baseline_v1.json), [privacy/toxicity](../data/eval/privacy_toxicity_baseline_v1.json).

## Historical operational diagnostics

Collection success: **29/30 (96.67%)**, excluding neither failures nor refusals from the collection denominator. A refusal is a collected response, not necessarily a successful answer to the question.

| Successful-case measurement | n | p50 seconds | p95 seconds |
| --- | ---: | ---: | ---: |
| Retrieval | 29 | 0.013572 | 0.017231 |
| Generation | 29 | 8.705171 | 20.332948 |
| Retrieval + generation stage sum | 29 | 8.719036 | 20.346065 |

These are historical diagnostics from a small collection, not repeated or load-tested latency benchmarks. Stage sums exclude initialization and other application overhead. Successful-case percentiles exclude the failed request. Token usage and retry counts were not recorded, so historical cost and retry rate are unknown. No p99 or service-level-objective claim is made.

Source: [operational_diagnostics_v1.json](../data/eval/operational_diagnostics_v1.json).

## Release interpretation

This release packages existing evidence into a navigable evaluation case study. It does not resolve the two unfinished contextual-relevance cases, pending human review, qualifier-sensitive judging, broad safety coverage, repeated performance benchmarking, or deployment monitoring.

For a new assistant release, the same pattern can preserve outputs, measure component and answer quality, inspect disagreements, and compare a controlled change against a baseline. Applying it elsewhere requires task-specific reviewed references, explicit policies and acceptance criteria; this dataset's scores and thresholds do not transfer automatically.
