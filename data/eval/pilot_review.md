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
