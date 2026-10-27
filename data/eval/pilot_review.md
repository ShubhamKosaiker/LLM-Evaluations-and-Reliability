# Pilot evaluation review

Automated report: data/eval/answer_eval_baseline_pilot_v2_2.json
Checkpoint commit: 482d700

Automated results: 17 correct, 1 incorrect, 2 judge errors.
These are provisional labels, not verified accuracy.

The observations below were proposed by the assistant after inspecting
answers and retrieved passages. They are awaiting human review.

## cough_04
The answer appears correct and supported across retrieved passages.
The judge assigned evidence quotes to incorrect passage IDs.
Keep the automated judge error recorded.

## headache_01
The answer appears correct and supported.
The judge altered a transcript quote and used internal ellipses.
Keep the automated judge error recorded.

## dyspnea_01
The automated judge marked the answer correct and supported.
However, the answer qualifies chest X-ray with "if available".
The retrieved passage applies that qualification to heart ultrasound.
Proposed assessment: partially correct with an unsupported qualification.

## dandruff_01
The retrieved passages are unrelated to the question.
The generator appropriately refused given that context.
The overall system failed to answer a corpus-answerable question.
Retrieval needs investigation; the exact cause is not yet established.

# Validation evaluation review

Automated report: data/eval/answer_eval_baseline_validation_v2_2.json
Automated results: 7 correct, 1 incorrect, 1 generation error, 1 judge error.
These are provisional labels, not verified accuracy.

The following assistant-proposed observations await human review.

## validation_acute_abdominal_01
Retrieval missed the passage defining Murphy's sign.
The generator appropriately refused given the supplied context.
The overall system failed to answer a corpus-answerable question.

## validation_headache_01
Generation failed with ValueError before producing an answer.
The saved error lacks enough detail to establish the cause.
Keep this execution failure separate from answer-quality scores.

## validation_chronic_abdominal_01
The gastric emptying scan answer appears supported by qfVHJQu-dfw_11.
The judge inserted internal ellipses into its evidence quote.
Keep the automated judge error recorded.

## validation_fever_01
The judge marked the answer correct but its faithfulness unsupported.
All five listed signs are supported by retrieved passage MIkO7oZZrtM_1.
Proposed review: supported; the automated faithfulness judgment is mistaken.
This demonstrates that valid quote formatting does not ensure judge accuracy.

## DeepEval faithfulness calibration — October 16, 2026

Baseline:
- DeepEval 4.2.8, Groq openai/gpt-oss-120b.
- 29 of 30 cases evaluated; one generation error skipped.
- All 29 evaluated cases received faithfulness = 1.0.
- These scores are uncalibrated automated judgments.

Controlled experiment:
- Case: headache_01.
- Intervention: added two deliberately unsupported claims
  about bacterial infection and emergency brain surgery.
- With penalize_ambiguous_claims=True: score = 0.667.
- With penalize_ambiguous_claims=False: score = 0.833.
- Both diagnostic runs extracted six claims:
  four supported, one borderline, one contradicted.
- An earlier corrupted-answer run scored 1.0;
  its intermediate verdicts were not captured.

Interpretation:
- Penalizing ambiguous claims increased sensitivity
  to unsupported statements in this example.
- The initial perfect score remains unexplained.
- These findings do not establish evaluator reliability.
- Do not overwrite the original DeepEval baseline.
- Next: calibrate against a small set of known-supported,
  unsupported, contradicted, and refusal cases.
## Answer relevance smoke findings

Report: data/eval/deepeval_answer_relevance_smoke_v1.json

- cough_01 scored 1.0: 2 yes verdicts and 10 borderline verdicts.
  DeepEval 4.2.8 counts borderline verdicts toward the score.
- dandruff_01 scored 1.0 despite providing only a refusal.
  The judge incorrectly explained that it answered the dandruff question.
- Preserve both automated scores. They are not verified answer quality.
  Answer relevance must be assessed alongside correctness, completeness
  and dedicated refusal evaluation.
- Both smoke cases completed. No baseline answers were regenerated.

## Answer relevance baseline (2026-10-23)

Report: data/eval/deepeval_answer_relevance_baseline_v1.json

- Completed 29 eligible answers: 20 pilot and 9 validation.
- validation_headache_01 was skipped because generation failed.
- Mean automated relevance score: pilot 1.0000; validation 0.8889.
- 28 answers scored 1.0; one scored 0.0.
- Scores agree with 107 statement verdicts:
  92 yes, 14 borderline, and 1 no.
- DeepEval 4.2.8 counts borderline verdicts toward relevance.
- Identical refusal text received 1.0 for dandruff_01 and 0.0 for
  validation_acute_abdominal_01, across different questions.
  Relevance scores alone cannot establish successful answering.
- Both smoke results were reused unchanged. The daily-token-limit
  interruption is retained in retry_history.
- These are automated observations, not verified accuracy.

## Contextual precision and recall smoke (2026-10-23)

Report: data/eval/deepeval_contextual_quality_smoke_v1.json

- Two cases completed across two metrics: four evaluations, no errors.
- cough_01: contextual precision 1.0; contextual recall 1.0.
- validation_acute_abdominal_01: both metrics scored 0.0.
- Score calculations and precision passage mappings were checked.
- The first cough passage lists red flags but does not explicitly state
  the X-ray indication. The judge inferred that connection.
  Preserve the automated precision score; its relevance judgment
  remains open to review.
- Recall assessed one reference sentence per case.
- References remain draft_assistant_reviewed. This two-case smoke test
  does not establish evaluator accuracy or dataset-wide performance.

## Contextual precision and recall baseline (2026-10-27)

Report: data/eval/deepeval_contextual_quality_baseline_v1.json

- Completed 60 metric evaluations across 30 saved retrieval cases.
- No pending evaluations or current evaluation errors.
- Pilot means: precision 0.8458; recall 0.9250.
- Validation means: precision 0.8500; recall 0.8667.
- Score calculations and passage mappings were checked.
- Four smoke evaluations were reused unchanged.
- Two daily-token-limit interruptions remain in retry_history.
- validation_chronic_abdominal_01 scored precision 0.5 and recall 1.0:
  the required information was present, but its passage ranked second.
- validation_hemoptysis_01 recall scored 0.6667 after the judge marked
  the standalone reference sentence "No." unsupported. This appears
  to be a reference-fragmentation/grading artifact requiring review.
- Preserve automated scores. Draft references and judge limitations
  prevent interpreting these results as verified accuracy.

## Application quality smoke evaluation

Report: data/eval/application_quality_smoke_v1.json

- Completed three saved-answer evaluations.
- cough_05: complete and clear, matching the proposed review labels.
- dyspnea_01: judged complete and clear, but missed the qualification
  "chest X-ray (if available)". Proposed review label remains partial:
  the source applies availability to heart ultrasound, not chest X-ray.
- dandruff_01: missing answer and clear refusal. Refusal was appropriate
  given unrelated retrieved passages; the overall system still failed.
- The file-access interruption and recovery are retained in the report.
- Preserve automated results. This smoke test exposed a qualifier-checking
  limitation and does not establish evaluator accuracy.
