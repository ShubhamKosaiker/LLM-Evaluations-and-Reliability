# LLM Evaluation & Reliability

An evidence-based evaluation project using a medical-video RAG assistant as the system under test. It measures retrieval and answer quality, investigates failures, and tests whether automated evaluators notice known mistakes.

**Key finding:** topic-preserving query wording improved pilot retrieval MRR but reduced validation MRR. Separately, an LLM judge scored an answer as correct while missing a qualifier that changed its meaning. Both the system and its evaluator need testing.

## Inspect the results in one command

Clone the repository and run from its root:

```powershell
python tools/review_results.py --check
```

Requires Python 3.11 or later and only the standard library. It reads committed reports, checks selected counts and score calculations, and prints the main results. It does **not** fetch videos, load an embedding model, generate answers, call a judge, or modify reports. No API key is needed.

Start with [Results and limitations](docs/RESULTS.md), [Setup and reproduction](docs/REPRODUCING.md), or the [artifact and command map](docs/ARTIFACTS.md).

## What was evaluated

- **Retrieval:** Hit@1/3/5 and MRR@5, query-wording and hybrid-search experiments.
- **RAG quality:** faithfulness, answer relevance, contextual relevance, contextual precision and recall.
- **Application quality:** source-relative correctness, completeness and clarity/style, with separate applicability rules for pure refusals and generation failures.
- **Evaluator reliability:** deliberately altered answers, claim-level verdicts, evidence checks and documented judge mistakes.
- **Safety smoke tests:** scope/refusal, context injection, synthetic-identifier extraction and an abusive-instruction request.
- **Historical operations:** collection success and successful-case retrieval/generation timings. Token usage and retry counts were not captured in these collections.

## Selected recorded results

These are development-set measurements and automated judgments, **not verified accuracy or clinical validation**.

| Measurement | Pilot | Validation |
| --- | ---: | ---: |
| Retrieval Hit@1 | 0.6000 (20 questions) | 0.8000 (10 questions) |
| Retrieval MRR@5 | 0.7833 | 0.8500 |
| Topic-wording experiment MRR@5 | 0.8417 | 0.8000 |
| Contextual precision | 0.8458 (20) | 0.8500 (10) |
| Contextual recall | 0.9250 (20) | 0.8667 (10) |
| GEval completeness | 0.9350 (20 answers) | 0.8556 (9 answers) |

The application run contains **85 numerical scores**, two not-applicable correctness slots for pure refusals, and three skipped metric slots for one generation failure. Historical answer collection succeeded for **29 of 30 questions**; collection success does not imply answer correctness.

Contextual relevance is **partial: 28/30 evaluated**, one error and one pending case. Perfect faithfulness/correctness scores coexist with documented judge failures; see the full report before interpreting those scores.

## System and evaluation design

The baseline uses 10 enabled educational medical videos and 70 transcript chunks. Chroma with `all-MiniLM-L6-v2` retrieves passages; Groq `openai/gpt-oss-20b` generates context-only answers. The judge is `openai/gpt-oss-120b`, using custom rubrics and DeepEval 4.2.8. Judge and generator therefore share a model family.

Saved collections record questions, draft references, retrieved passages, generated answers and errors. Later evaluators score those saved answers instead of silently regenerating them. Reports retain per-case explanations, configurations and, where implemented, hashes and retry history.

The workflow is inspired by [CampusX's RAG evaluation project](https://github.com/campusx-official/rag-eval-deepeval). The corpus, judge, rubrics and safety checks differ; scores are not directly comparable. Safety marker checks here are not DeepEval PII or toxicity metric scores.

## Repository guide

| Location | Purpose |
| --- | --- |
| `src/` | Existing RAG, collection, evaluation and experiment entry points |
| `data/videos/videos.csv` | Enabled source videos and attribution |
| `data/eval/` | Frozen collections, draft annotations, reports and review notes |
| `docs/` | Consolidated results, reproduction instructions and navigation |
| `requirements/` | Separate application and evaluation dependency definitions |
| `tools/review_results.py` | Read-only, offline evidence review |
| `tests/test_review_results.py` | Offline release-check tests |

Historical scripts and reports retain their paths: moving them would affect imports, fixed output paths and provenance hashes. Navigate through the artifact map rather than treating the directory as a list of equally important files.

## Reproduction boundary

A fresh clone can inspect and check the committed evidence offline. **Exact retrieval replay is not self-contained:** original transcript/chunk files and the Chroma database are ignored and are not included. Refetching captions may yield different text or fail timestamp-based corrections. Live LLM responses can also change.

Dependency files pin directly observed versions; they are not complete transitive lockfiles. Original work was performed on Windows with Python 3.11.2 and separate application/evaluation environments. See [reproduction details](docs/REPRODUCING.md) before installing or rerunning anything.

## Practical use and current limits

The same evaluation pattern can compare updates to an internal document or support assistant: inspect retrieved evidence, measure answer quality, and review regressions before release. This repository demonstrates that workflow on a small development corpus; it is not a deployed monitoring service or a validated medical product.

Human adjudication, reliable qualifier-sensitive judging, broad adversarial testing, repeated operational benchmarks and online monitoring remain future work. The existing validation questions were used during development and are not an untouched final test set. No deployment-readiness or general safety claim is made.
