# Packaging validation

Prepared against the public repository at `2b15033c88ccc3643790d75aa80a1929e0168081` on 4 November 2026. This records checks on the packaging work; it is not a new evaluation run or a claim that this package has been committed or deployed.

## Executed locally

- `python tools/review_results.py --check` against the fetched checkpoint: passed saved-count, selected score/mean and historical timing consistency checks.
- `python -m unittest discover -s tests -v`: six tests passed, including detection of altered retrieval rank, GEval raw score, application mean and historical latency, and visibility of partial contextual-relevance coverage.
- All local Markdown links in README and the documentation resolved against the checkpoint plus package.
- All 18 existing Python source files parsed successfully.
- Existing ranking scorer self-test passed: six ranking fixtures and two invalid-input checks.
- Four documented custom-judge / safety collector `--help` commands executed successfully without third-party SDKs or model calls.

No source files under `src/`, saved JSON reports, fixtures or review observations are changed by this package. No generator or judge requests were made, no corpus/index was rebuilt, no dependency installation was performed, and no commit or push was created.

## Not established by these checks

- Clean installation of every pinned application/evaluation dependency and its transitive dependencies.
- Execution of DeepEval CLI help/preview in a newly installed environment. Those commands were checked against the source; strict provenance guards may reject old reports in a different checkout.
- Exact replay of the ignored transcript/chunk/index snapshot.
- Windows-specific file locking behaviour or a current Groq quota/model permission check.
- Human-verified labels, clinical correctness, a comprehensive safety benchmark, repeated production latency, retry rates, cost or deployment readiness.

The requirement files pin directly observed versions, not an invented complete historical lockfile. The optional transcript dependency is intentionally unpinned because its historical version was not captured.
