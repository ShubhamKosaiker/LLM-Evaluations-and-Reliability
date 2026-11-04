# Artifact and command map

Start with [RESULTS.md](RESULTS.md). Raw reports retain their original locations for provenance. This map groups them by purpose so a reviewer need not inspect the entire `data/eval/` directory.

## Core evidence

| Purpose | Files |
| --- | --- |
| Source list | [videos.csv](../data/videos/videos.csv) |
| Draft references and relevant chunk labels | [pilot goldens](../data/eval/retrieval_goldens.json), [validation goldens](../data/eval/retrieval_validation_goldens.json) |
| Saved RAG answers, contexts and errors | [pilot](../data/eval/rag_baseline_pilot.json), [validation](../data/eval/rag_baseline_validation.json) |
| Retrieval baseline | [pilot](../data/eval/retrieval_baseline.json), [validation](../data/eval/retrieval_validation_baseline.json) |
| Faithfulness | [baseline](../data/eval/deepeval_faithfulness_baseline_v1.json), [calibration fixture](../data/eval/faithfulness_calibration.json), [calibration results](../data/eval/deepeval_faithfulness_calibration_v1.json) |
| Contextual precision/recall | [baseline](../data/eval/deepeval_contextual_quality_baseline_v1.json) |
| Contextual relevance, partial | [baseline](../data/eval/deepeval_contextual_relevance_baseline_v1.json) |
| Answer relevance | [baseline](../data/eval/deepeval_answer_relevance_baseline_v1.json) |
| Numerical application metrics | [baseline](../data/eval/deepeval_application_baseline_v1.json), [smoke results](../data/eval/deepeval_application_smoke_v1.json) |
| Historical collection/timing diagnostics | [diagnostics](../data/eval/operational_diagnostics_v1.json) |
| Per-case interpretation and unresolved findings | [review notes](../data/eval/pilot_review.md) |

## Controlled experiments and evaluator checks

| Experiment | Files / entry point |
| --- | --- |
| Topic-preserving wording comparison | [pilot](../data/eval/retrieval_query_topic.json), [validation](../data/eval/retrieval_validation_topic.json); `src/experiment_query_topic.py`, `src/run_retrieval_validation.py topic` |
| Prefix removal | [report](../data/eval/retrieval_query_wording.json); `src/experiment_query_wording.py` |
| Manual short queries | [pilot](../data/eval/retrieval_query_short.json), [validation](../data/eval/retrieval_validation_query_short.json); `src/experiment_query_short.py` |
| Dense/BM25 hybrid | [pilot](../data/eval/retrieval_hybrid.json), [validation](../data/eval/retrieval_validation_hybrid.json); `src/experiment_hybrid.py` |
| Stemmed hybrid | [pilot](../data/eval/retrieval_hybrid_stemmed.json), [validation](../data/eval/retrieval_validation_hybrid_stemmed.json); `src/experiment_hybrid_stemmed.py` |
| Custom reference-relative correctness + isolated faithfulness | [pilot](../data/eval/answer_eval_baseline_pilot_v2_2.json), [validation](../data/eval/answer_eval_baseline_validation_v2_2.json); `src/evaluate_rag_answers.py` |
| Custom completeness/clarity/refusal sample | [report](../data/eval/application_quality_smoke_v1.json); `src/evaluate_rag_answers.py --application-smoke` |
| Qualifier-sensitive controlled pair | [fixture](../data/eval/application_qualifier_calibration.json), [report](../data/eval/application_qualifier_calibration_v1.json); `src/evaluate_rag_answers.py --qualifier-calibration` |

The original retrieval remains the default. Development experiments and custom rubric labels are not additional production features or directly comparable numerical GEval scores.

## Safety fixtures and responses

| Suite | Fixture | Saved responses | Collector flag |
| --- | --- | --- | --- |
| Scope/refusal | [goldens](../data/eval/scope_safety_goldens.json) | [report](../data/eval/scope_safety_baseline_v1.json) | `--scope-smoke` |
| Context injection | [goldens](../data/eval/context_injection_goldens.json) | [report](../data/eval/context_injection_baseline_v1.json) | `--context-injection` |
| Synthetic privacy/toxicity | [goldens](../data/eval/privacy_toxicity_goldens.json) | [report](../data/eval/privacy_toxicity_baseline_v1.json) | `--privacy-toxicity` |

These flags belong to `src/collect_rag_answers.py`, not the DeepEval runner. They support `--preview`, `--resume` and `--max-cases`. Existing report identity guards still apply. Exact marker/refusal checks are diagnostics; semantic review remains separate.

## Entry points

| Task | Interpreter / command | Needs |
| --- | --- | --- |
| Read/check saved evidence | `python tools/review_results.py --check` | Standard library; no key |
| Offline release tests | `python -m unittest discover -s tests -v` | Standard library; no key |
| Ranking scorer self-test | `python src/eval_retriever.py --self-test` | Standard library; no live retrieval |
| Index against local chunks | `.venv` → `python src/check_index.py` | Original local chunks and Chroma index |
| Original RAG example | `.venv` → `python src/rag_pipeline.py` | Original index, embedding model and Groq key; live request |
| Baseline retrieval evaluation | `.venv` → `python src/eval_retriever.py` | Matching original chunks/index; refuses to overwrite an existing report |
| Validation retrieval | `.venv` → `python src/run_retrieval_validation.py baseline` | Matching original chunks/index; preserve existing outputs |
| Saved-answer DeepEval metrics | `.venv-eval` → `python -m src.run_deepeval_faithfulness --help` | DeepEval environment; key needed only for live judging |
| Custom saved-answer judge | `.venv-eval` → `python src/evaluate_rag_answers.py --help` | Groq environment; key for live judging |
| Frozen-context safety collection | `.venv-eval` → `python src/collect_rag_answers.py --scope-smoke --help` | Key for live generation; no live retriever |

DeepEval runner modes: `baseline` (faithfulness), `calibration`, `contextual-relevance-smoke`, `contextual-relevance-baseline`, `answer-relevance-smoke`, `answer-relevance-baseline`, `contextual-quality-smoke`, `contextual-quality-baseline`, `application-smoke`, `application-baseline`.

Relevance/context-quality/application modes support `--preview`. Application and contextual-quality baseline runs use `--max-metrics` for a budget of new metric jobs; relevance runs use `--max-cases`. Do not assume the faithfulness `baseline`/`calibration` modes support the same flags or guards. Read `--help` and the source before intentional reruns.

## Organisation policy

- `docs/` is the narrative entry point; `data/eval/` is the evidence archive.
- Keep historical report/source paths stable. A move can invalidate imports, source references and raw-byte hashes even when the algorithm is unchanged.
- New disposable local experiments can go in ignored `data/local_runs/`; final evidence should have an explicit versioned path and documentation.
- The ignored virtual environments are not source code. Older untracked debugging files and the local `tatus --short` item are not release deliverables; inspect them separately before any cleanup. This packaging change neither deletes nor stages them.
- `PROJECT_CONTEXT.md` is a legacy historical note with outdated milestones, not the current setup/results specification. Current technical entry points are README and these documents. Preserve local edits until reviewed separately.

The reference for coverage is [CampusX rag-eval-deepeval](https://github.com/campusx-official/rag-eval-deepeval). This release includes a narrower safety/operational demonstration and explicit evaluator caveats; it does not implement all of that project's live monitoring, budget, or regression-gate functionality.
