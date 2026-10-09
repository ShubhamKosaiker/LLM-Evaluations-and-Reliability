# LLM Evaluations and Reliability: project handoff

Last updated: 9 October 2026 (Asia/Calcutta).
Last user-confirmed code checkpoint: `35007dd`, pushed to `main`; working tree clean.
Repository: https://github.com/ShubhamKosaiker/LLM-Evaluations-and-Reliability

## Read this before continuing in a new chat

This is the agreed project direction. Recover the current repository state, read
this file, and continue the next incomplete milestone. Do not restart the project
or resume retrieval tuning simply because it was the most recent experiment.
Update this handoff when a milestone, decision, or checkpoint changes.
If later explicit user instructions change the scope, follow them and record the
change here. Do not treat a clarification or conceptual question as a new project.

## Purpose and scope

The main product is an evaluation pipeline. The medical-video RAG assistant is
the system under test. The goal is a clean, reproducible portfolio project that
shows evaluation skills relevant to evals, reliability, and research engineering
roles, including Callosum and AISI. No promise of a hiring outcome.

The user wants a clear CampusX-style RAG evaluation workflow, referring to
https://github.com/campusx-official/rag-eval-deepeval . That reference is an
inspiration, not a requirement to copy every component or an already-verified
implementation in this repository.

Prioritize test design, evaluator reliability, failure analysis, controlled
comparisons, and readable evidence. Do not let chatbot features or repeated
retrieval experiments dominate the evaluation project.

## Working agreement

- Explain each step simply: what it does, why it matters, and what output to expect.
- Give small batches of concrete Windows PowerShell instructions, normally no more
  than five steps. The user runs commands and returns results.
- Do not claim a feature is implemented until it exists in the repository and has
  been tested. Distinguish prepared code, local checks, user-run results, and
  confirmed Git checkpoints.
- Keep the project organized. Prefer one configurable evaluation workflow over
  a growing collection of separate scripts. Consolidate carefully after reading
  current imports; preserve historical reports and provenance.
- Preserve the baseline. No retrieval changes have been adopted in the main
  assistant. Do not silently swap in the best-scoring development experiment.
- Review failures and regressions; do not optimize only aggregate scores.
- Maintain this file for future chats so the user need not repeat the plan.

## Verified progress at the checkpoint

- Baseline RAG pipeline exists: transcripts, cleaning, chunking, Chroma retrieval,
  and Groq answer generation.
- Current corpus: 10 enabled videos, 70 chunks. An index check passed with zero
  missing/extra IDs or text mismatches.
- Retriever: `all-MiniLM-L6-v2`, Chroma collection `medical_videos` under
  `data/chroma_db`. `retrieve` defaults to 3 passages; retrieval evaluation uses 5.
- Generator: `src/generator.py`, Groq model `openai/gpt-oss-20b`, temperature 0,
  context-only prompt and an insufficient-information instruction. This prompt
  is not evidence that refusals or answer quality have been evaluated.
- Initial 20 questions and additional 10 questions have source-relative draft
  answers, supporting chunk IDs, and evidence excerpts.
- Index and retrieval evaluation scripts and reports are committed.
- Root `README.md` and `requirements.txt` were not found when checked at `35007dd`.
  Confirm the actual current layout before creating or moving files.

## Retrieval investigation: closed for now

Baseline, query-prefix removal, topic-preserving wording, manual short queries,
plain hybrid retrieval, and stemmed hybrid retrieval have been run. Keep their
reports as evidence, but pause further retrieval tuning.

| Dataset and method | Hit@1 | Hit@3 | Hit@5 | MRR@5 |
| --- | ---: | ---: | ---: | ---: |
| Initial 20 baseline | 0.60 | 1.00 | 1.00 | 0.7833 |
| Initial 20 stemmed hybrid | 0.85 | 0.95 | 1.00 | 0.9125 |
| Additional 10 baseline | 0.80 | 0.90 | 0.90 | 0.8500 |
| Additional 10 stemmed hybrid | 0.80 | 0.90 | 1.00 | 0.8583 |

Stemmed hybrid is a promising development candidate, not an adopted default.
It combines dense and BM25 keyword candidates using equal-weight reciprocal rank
fusion: 20 candidates per method, RRF constant 60, BM25 k1 1.2 and b 0.75.
English Snowball stemming requires `nltk==3.9.2` and retains stopwords.

Examples of observed failures and tradeoffs:
- Murphy's sign ranked 12 with the original query and 1 with the term alone.
  Plain hybrid placed its answer 3rd; stemmed hybrid placed it 4th.
- The dehydration question uses 'dehydration', while its passage uses
  'dehydrated'. Plain keyword search ranked that passage 39th; plain hybrid
  dropped it from the top five. Stemmed hybrid recovered it at rank 3, compared
  with rank 1 in the original dense baseline.
- An earlier short-query run failed an exact baseline-ranking check on two fever
  questions. Isolated checks and a later complete run reproduced the baselines.
  The earlier mismatch remains unexplained; do not claim a root cause or remove
  the reproducibility guard merely to obtain a successful run.

These are small, assistant-drafted datasets needing human review. The additional
10 questions have now been inspected and used in development; they are no longer
an untouched holdout. Future generalization claims require fresh test questions.

## Agreed next milestones, in order

1. Inventory and organize the existing repository. Add a concise README and
   dependency instructions. Consolidate experiment entry points without changing
   the default retrieval behavior or erasing prior reports. Keep cleanup bounded.
2. Build one end-to-end evaluation runner using the original retriever. Save each
   question, reference answer, retrieved IDs/text, generated answer, configuration,
   and run provenance. Capture generation errors rather than treating them as
   correct refusals. First collect outputs before assigning quality scores.
3. Add answer correctness and faithfulness checks with explicit rubrics. Choose
   and verify a consistent framework such as DeepEval; it is not yet integrated.
   Add relevance and refusal/insufficient-evidence cases progressively.
4. Assess evaluator reliability against human review. Inspect disagreements and
   distinguish retrieval, generation, dataset-label, and evaluator failures.
5. Publish a readable evaluation report with failure examples, controlled
   comparisons, limitations, and reproducible commands. Add fresh questions before
   making claims about generalization. Robustness and latency can follow when the
   core answer evaluation works.

The next milestone is not another retriever experiment. It is a clean end-to-end
record: question -> retrieved passages -> generated answer -> evaluation results.

## Immediate next action

Await the output of these commands already requested from the user:

```powershell
git ls-files
python -c "from importlib.metadata import version; names=['chromadb','sentence-transformers','groq','python-dotenv','nltk']; print('\n'.join(f'{n}=={version(n)}' for n in names))"
```

Use the output to prepare the bounded organization/documentation step, then the
end-to-end answer collection runner. Do not ask the user to repeat the project
concept or restart completed retrieval work.
