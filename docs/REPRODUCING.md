# Setup, evidence inspection and reproduction

Run commands from the repository root. The primary supported workflow for this release is **offline inspection of saved evidence**. Live replay is optional and has additional data, environment and provenance requirements.

## 1. Inspect the committed evidence without credentials

Python 3.11 or later is sufficient:

```powershell
python tools/review_results.py --check
python -m unittest discover -s tests -v
```

The review command recomputes selected retrieval scores, checks collection counts and saved GEval normalization, validates selected aggregate means, and checks historical timing arithmetic. It reports contextual relevance as partial instead of turning unfinished cases into zero scores. Neither command imports the RAG application or calls an LLM.

Expected: a saved-report consistency `PASS`, explicit application-score denominators, and a partial-context-relevance notice. A `PASS` does not certify the judge's semantic decisions, corpus identity, safety, or medical correctness. These checks do not compare historical byte-level provenance hashes with today's checkout.

## 2. Optional environments for live work

The recorded workflow used Windows PowerShell, Python 3.11.2 and two environments. Direct dependency versions below are supported by recorded reports or the implementation history; they are **not a fully locked transitive environment**. The complete original freeze, embedding weights revision and corpus-generation dependency version were not preserved. A clean end-to-end installation from these new requirement files has not been benchmarked as part of this packaging release.

For a fresh installation, with Python 3.11 available:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements/app.txt
.\.venv\Scripts\python.exe -m pip check

py -3.11 -m venv .venv-eval
.\.venv-eval\Scripts\python.exe -m pip install -r requirements/eval.txt
.\.venv-eval\Scripts\python.exe -m pip check
```

Existing working environments should be reused rather than recreated. Keep DeepEval separate from the application environment: an earlier combined installation produced dependency conflicts. Calling each environment's Python directly avoids activation-policy changes.

The optional transcript-fetch helper is listed separately in `requirements/corpus.txt`; its original version is unknown. Installing it now creates a new corpus-build environment, not a reconstruction of the historical one.

For deliberate live requests, copy the template and set a key locally:

```powershell
Copy-Item .env.example .env
```

Set `GROQ_API_KEY` in `.env`. Do not commit the key. Historical model IDs are `openai/gpt-oss-20b` for generation and `openai/gpt-oss-120b` for judging. Confirm account permissions, current availability and quotas in the [Groq console](https://console.groq.com/) before live work. Do not silently substitute a model in a baseline experiment.

## 3. Data required for retrieval replay

The clone includes video URLs, draft goldens, saved answer contexts and evaluation reports. It does not include these ignored local directories:

- `data/transcripts/`
- `data/cleaned_transcripts/`
- `data/chunks/`
- `data/chroma_db/`

To reproduce the original retrieval snapshot, restore the original local chunks/index and validate them against the golden corpus hash. If the original transcripts/chunks are unavailable, exact retrieval replay is currently unavailable. The saved reports still support offline arithmetic and failure inspection.

The original corpus-build order is:

```powershell
.\.venv\Scripts\python.exe src/transcript_fetcher.py
.\.venv\Scripts\python.exe src/transcript_cleaner.py
.\.venv\Scripts\python.exe src/chunker.py
.\.venv\Scripts\python.exe src/retriever.py
.\.venv\Scripts\python.exe src/check_index.py
```

**These are live rebuild commands, not part of the offline quick start.** Fetching reaches YouTube, indexing downloads embedding weights if missing, and the scripts write local corpus/index files. Use a separate checkout when building a new corpus. Caption edits, unavailable videos, timestamp corrections and model/library revisions can change the snapshot. Do not replace the recorded 70-chunk baseline with newly fetched data and call it the same corpus.

`check_index.py` checks stored records against local chunks. The retrieval evaluator additionally checks the chunk count/text hash against its golden snapshot. Those are different checks; matching the local index alone does not establish a match to the historical baseline.

## 4. Inspect command interfaces

These commands display interfaces rather than running evaluations:

```powershell
.\.venv-eval\Scripts\python.exe -m src.run_deepeval_faithfulness --help
.\.venv-eval\Scripts\python.exe src/evaluate_rag_answers.py --help
.\.venv-eval\Scripts\python.exe src/collect_rag_answers.py --scope-smoke --help
.\.venv-eval\Scripts\python.exe src/collect_rag_answers.py --context-injection --help
.\.venv-eval\Scripts\python.exe src/collect_rag_answers.py --privacy-toxicity --help
```

The saved-answer collector and custom judge have several explicit workflows; see [ARTIFACTS.md](ARTIFACTS.md). Do not assume all modules support `python -m src...`: some original modules use script-relative imports and should be invoked as `python src/<file>.py`.

## 5. Preview and provenance guards

For example, the numerical application workflow supports:

```powershell
.\.venv-eval\Scripts\python.exe -m src.run_deepeval_faithfulness --mode application-baseline --preview
```

The preview makes no judge requests or report writes. It may intentionally fail if report identity no longer matches the source files, runner, adapter, packages or configuration. `--preview` is not a bypass for provenance validation. A current source file may differ from the one used for an older report; Windows CRLF versus LF can also change raw SHA-256 hashes while leaving parsed JSON unchanged.

For old runs, inspect the report's recorded hashes and file history (`git log --oneline -- <path>`). Use the matching source revision and original file bytes for an exact continuation. If those bytes are unavailable, describe the operation as a new run rather than changing the historical report's identity to force a match. The offline results reviewer does not require matching runner hashes because it does not resume an evaluation.

Completed reports must remain unchanged. Fixed-output workflows are not general-purpose output-override tools. Create an isolated experiment with explicit new output paths and provenance before running a changed evaluator. The baseline collection command has an `--output` argument, but many DeepEval workflows use fixed report constants.

## Live-run operating rules

- Preserve the original generator, retriever, prompts, source snapshots and reports before changes.
- Keep the configured 60-second request pacing. Pacing does not guarantee quota availability; SDK retries and account limits are separate concerns. The original pilot/validation collector does not provide the same paced safety-workflow interface; do not use it for an unattended bulk rerun.
- Do not modify evaluators during a run. Metric-level checkpoints skip completed compatible work, but some stored errors require review before retrying.
- On `PermissionError`, close competing writers/sync locks and inspect the saved checkpoint; do not delete historical output. Bounded retries exist in some workflows, not every historical script.
- Distinguish API/rate-limit errors, generation failures, appropriate refusals, low quality scores and judge errors. Their denominators differ.
- Capture exact environment versions and model metadata for any future run. New model responses or scores need not equal the historical ones.

No packages, models or transcripts need to be installed or fetched merely to review this release's evidence.
