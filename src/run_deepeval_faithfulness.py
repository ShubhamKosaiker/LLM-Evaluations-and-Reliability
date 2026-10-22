
"""DeepEval faithfulness evaluation for saved RAG answers.

Modes:
    baseline     - Existing pilot and validation evaluations.
    calibration  - Five controlled evaluator-reliability cases.

Does not rerun retrieval or answer generation.
"""

import argparse
import hashlib
import json
import math
from importlib.metadata import version
from datetime import datetime, timezone
from pathlib import Path

from deepeval.metrics import FaithfulnessMetric, ContextualRelevancyMetric
from deepeval.test_case import LLMTestCase

from src.deepeval_groq_adapter import GroqDeepEvalJudge


SOURCES = [
    "data/eval/rag_baseline_pilot.json",
    "data/eval/rag_baseline_validation.json",
]

OUTPUT = Path("data/eval/deepeval_faithfulness_baseline_v1.json")

CALIBRATION_INPUT = Path("data/eval/faithfulness_calibration.json")
CALIBRATION_OUTPUT = Path(
    "data/eval/deepeval_faithfulness_calibration_v1.json"
)

THRESHOLD = 0.7
REQUEST_INTERVAL = 60


def save(report, output=OUTPUT):
    """Atomically save progress without overwriting another report."""
    report["updated_at_utc"] = datetime.now(timezone.utc).isoformat()

    temp = output.with_suffix(".tmp")
    temp.write_text(
        json.dumps(report, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    temp.replace(output)


def file_sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_calibration():
    """Evaluate controlled variants with checkpointing and resume guards."""

    data = json.loads(
        CALIBRATION_INPUT.read_text(encoding="utf-8")
    )

    input_hash = file_sha256(CALIBRATION_INPUT)

    # Ensure the underlying saved RAG baseline has not changed.
    source_path = Path(data["source_report"])
    if file_sha256(source_path) != data["source_report_sha256"]:
        raise ValueError("Original source report has changed.")

    variants = data["variants"]
    variant_ids = [variant["id"] for variant in variants]

    if len(variants) != 5 or len(set(variant_ids)) != 5:
        raise ValueError("Expected exactly five unique variants.")

    if variant_ids != ["A", "B", "C", "D", "E"]:
        raise ValueError("Unexpected calibration variant IDs.")

    # Configuration is fixed for this calibration experiment.
    configuration = {
        "model": "openai/gpt-oss-120b",
        "threshold": THRESHOLD,
        "penalize_ambiguous_claims": True,
        "async_mode": False,
        "include_reason": True,
        "request_interval_seconds": REQUEST_INTERVAL,
    }

    if CALIBRATION_OUTPUT.exists():
        report = json.loads(
            CALIBRATION_OUTPUT.read_text(encoding="utf-8")
        )

        if report.get("input_sha256") != input_hash:
            raise ValueError(
                "Calibration input changed. Refusing to resume."
            )

        if report.get("configuration") != configuration:
            raise ValueError(
                "Calibration configuration changed. Refusing to resume."
            )

        if report.get("source_report_sha256") != data["source_report_sha256"]:
            raise ValueError(
                "Source report identity changed. Refusing to resume."
            )

        print("Resuming calibration", flush=True)

    else:
        report = {
            "metric": "deepeval_faithfulness_calibration",
            "source_case_id": data["source_case_id"],
            "input_sha256": input_hash,
            "source_report_sha256": data["source_report_sha256"],
            "configuration": configuration,
            "expectation_status": data.get("expectation_status"),
            "results": {},
        }

    judge = GroqDeepEvalJudge(
        model=configuration["model"],
        interval=REQUEST_INTERVAL,
    )

    for variant in variants:
        variant_id = variant["id"]

        if report["results"].get(variant_id, {}).get("status") == "evaluated":
            print("Skipping completed variant:", variant_id)
            continue

        print("Evaluating variant:", variant_id, flush=True)

        metric = FaithfulnessMetric(
            model=judge,
            threshold=THRESHOLD,
            include_reason=True,
            async_mode=False,
            penalize_ambiguous_claims=True,
        )

        test_case = LLMTestCase(
            input=data["query"],
            actual_output=variant["answer"],
            retrieval_context=data["contexts"],
        )

        try:
            metric.measure(test_case)

            result = {
                "status": "evaluated",
                "score": metric.score,
                "reason": metric.reason,
                "passed": metric.is_successful(),
                "claims": list(metric.claims or []),
                "verdicts": [
                    {
                        "verdict": (
                            v.verdict.value
                            if hasattr(v.verdict, "value")
                            else str(v.verdict)
                        ),
                        "reason": v.reason,
                    }
                    for v in (metric.verdicts or [])
                ],
            }

        except Exception as exc:
            result = {
                "status": "evaluation_error",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }

        report["results"][variant_id] = result
        save(report, output=CALIBRATION_OUTPUT)

        print(
            variant_id,
            result["status"],
            result.get("score", "-"),
            flush=True,
        )

        # Stop on error rather than repeatedly hitting the API.
        if result["status"] == "evaluation_error":
            print("Stopped after evaluation error.", flush=True)
            break

    print("Calibration report:", CALIBRATION_OUTPUT, flush=True)


def run_baseline():
    """Preserve the original baseline evaluation workflow."""

    if OUTPUT.exists():
        report = json.loads(OUTPUT.read_text(encoding="utf-8"))

        if report.get("sources") != SOURCES:
            raise ValueError(
                "Existing baseline report uses different input sources."
            )

        print("Resuming existing baseline report", flush=True)

    else:
        report = {
            "metric": "deepeval_faithfulness",
            "threshold": 0.7,
            "model": "openai/gpt-oss-120b",
            "sources": SOURCES,
            "results": {},
        }

    judge = GroqDeepEvalJudge(interval=60)

    for source in SOURCES:
        dataset = Path(source).stem

        data = json.loads(
            Path(source).read_text(encoding="utf-8-sig")
        )

        for case in data["cases"]:
            case_id = case["id"]
            key = f"{dataset}/{case_id}"

            if key in report["results"]:
                continue

            print(f"Evaluating {key}", flush=True)

            if case["status"] != "success":
                report["results"][key] = {
                    "status": "skipped_generation_error",
                    "source_status": case["status"],
                }
                save(report)
                continue

            test_case = LLMTestCase(
                input=case["query"],
                actual_output=case["answer"],
                retrieval_context=case["contexts"],
            )

            metric = FaithfulnessMetric(
                model=judge,
                threshold=0.7,
                include_reason=True,
                async_mode=False,
            )

            try:
                metric.measure(test_case)

                result = {
                    "status": "evaluated",
                    "score": metric.score,
                    "reason": metric.reason,
                    "passed": metric.is_successful(),
                }

            except Exception as exc:
                result = {
                    "status": "evaluation_error",
                    "error_type": type(exc).__name__,
                    "error": str(exc),
                }

            report["results"][key] = result
            save(report)

            print(
                f"{case_id}: {result['status']}",
                flush=True,
            )

    print("Baseline report:", OUTPUT, flush=True)



CONTEXTUAL_CASE_IDS = ("cough_01", "dandruff_01")
CONTEXTUAL_OUTPUT = Path(
    "data/eval/deepeval_contextual_relevance_smoke_v1.json"
)


def contextual_cases():
    """Load only saved queries, contexts and IDs; never reference answers."""
    source = Path(SOURCES[0])
    data = json.loads(source.read_text(encoding="utf-8-sig"))
    selected = []
    for case_id in CONTEXTUAL_CASE_IDS:
        matches = [c for c in data["cases"] if c["id"] == case_id]
        if len(matches) != 1:
            raise ValueError(f"Expected one saved case: {case_id}")
        case = matches[0]
        contexts, ids = case["contexts"], case["retrieved_ids"]
        if (not contexts or len(ids) != len(contexts)
                or len(set(ids)) != len(ids)
                or any(not isinstance(c, str) or not c.strip() for c in contexts)
                or not isinstance(case["query"], str) or not case["query"].strip()):
            raise ValueError(f"Invalid saved retrieval data: {case_id}")
        selected.append({
            "id": case_id, "query": case["query"],
            "contexts": contexts, "retrieved_ids": ids,
        })
    return selected


def contextual_verdicts(metric, case):
    """Preserve DeepEval 4.2.8's ordered per-passage statement judgments."""
    groups = metric.verdicts_list
    if len(groups) != len(case["contexts"]):
        raise ValueError("Verdict groups do not match the saved passages")
    passages = []
    for index, (cid, group) in enumerate(zip(case["retrieved_ids"], groups)):
        checks = []
        for v in group.verdicts:
            verdict = v.verdict.value if hasattr(v.verdict, "value") else str(v.verdict)
            if verdict not in ("yes", "no") or not v.statement.strip():
                raise ValueError("Invalid statement-level relevance verdict")
            checks.append({
                "statement": v.statement, "verdict": verdict, "reason": v.reason,
            })
        passages.append({
            "context_index": index, "context_id": cid,
            "statement_verdicts": checks,
        })
    return passages


def run_contextual_smoke(preview=False):
    """Two fixed retrieval cases; no retrieval or generation is rerun."""
    cases = contextual_cases()
    packages = {name: version(name) for name in ("deepeval", "groq", "pydantic")}
    if packages["deepeval"] != "4.2.8":
        raise ValueError("This experiment requires DeepEval 4.2.8")
    configuration = {
        "metric": "ContextualRelevancyMetric",
        "model": "openai/gpt-oss-120b", "threshold": THRESHOLD,
        "threshold_status": "provisional_not_a_validated_quality_gate",
        "eval_mode": "llm", "async_mode": False, "strict_mode": False,
        "include_reason": True, "request_interval_seconds": REQUEST_INTERVAL,
    }
    print("Contextual relevance smoke: saved retrieval only", flush=True)
    for case in cases:
        print(f"{case['id']}: {len(case['contexts'])} passages", flush=True)
        print("  Query:", case["query"], flush=True)
        print("  IDs:", ", ".join(case["retrieved_ids"]), flush=True)
    print("Output:", CONTEXTUAL_OUTPUT, flush=True)
    if preview:
        print("PREVIEW: no judge client, API calls or report writes.", flush=True)
        return

    identity = {
        "schema_version": 1,
        "run_type": "contextual_relevance_smoke",
        "source": SOURCES[0],
        "source_sha256": file_sha256(Path(SOURCES[0])),
        "runner_sha256": file_sha256(Path(__file__)),
        "adapter_sha256": file_sha256(
            Path(__file__).with_name("deepeval_groq_adapter.py")
        ),
        "configuration": configuration, "package_versions": packages,
        "selected_case_ids": list(CONTEXTUAL_CASE_IDS),
        "cases": cases,
    }
    if CONTEXTUAL_OUTPUT.exists():
        report = json.loads(CONTEXTUAL_OUTPUT.read_text(encoding="utf-8"))
        if any(report.get(k) != value for k, value in identity.items()):
            raise ValueError("Inputs, code or configuration changed; refusing to resume")
        results = report.get("results", {})
        if not isinstance(results, dict) or set(results) - set(CONTEXTUAL_CASE_IDS):
            raise ValueError("Unexpected results in existing smoke report")
        if any(r.get("status") != "evaluated" for r in results.values()):
            raise ValueError("Prior evaluation error needs review; no automatic retry")
    else:
        report = {**identity, "results": {}, "review_status": "automated_unreviewed"}
    report["run_status"] = "in_progress"
    save(report, CONTEXTUAL_OUTPUT)

    judge = None
    for case in cases:
        case_id = case["id"]
        if case_id in report["results"]:
            print("Skipping completed case:", case_id, flush=True)
            continue
        try:
            if judge is None:
                judge = GroqDeepEvalJudge(
                    model=configuration["model"], interval=REQUEST_INTERVAL
                )
            metric = ContextualRelevancyMetric(
                model=judge, threshold=THRESHOLD, eval_mode="llm",
                include_reason=True, async_mode=False, strict_mode=False,
            )
            # Neither generated answers nor expected answers enter this test case.
            metric.measure(LLMTestCase(
                input=case["query"], retrieval_context=case["contexts"]
            ))
            passages = contextual_verdicts(metric, case)
            score = metric.score
            if not isinstance(score, (int, float)) or not math.isfinite(score) or not 0 <= score <= 1:
                raise ValueError("Invalid contextual relevance score")
            result = {
                "status": "evaluated", "score": score,
                "reason": metric.reason, "passed": metric.is_successful(),
                "passages": passages,
            }
            report["results"][case_id] = result
            save(report, CONTEXTUAL_OUTPUT)
            print(f"{case_id}: score={score:.4f}; provisional_pass={result['passed']}", flush=True)
            print("  Why:", metric.reason, flush=True)
        except Exception as exc:
            report["results"][case_id] = {
                "status": "evaluation_error", "error_type": type(exc).__name__,
            }
            report["run_status"] = "stopped_evaluation_error"
            save(report, CONTEXTUAL_OUTPUT)
            print(f"{case_id}: EVALUATION ERROR ({type(exc).__name__}); stopped.", flush=True)
            return
        except KeyboardInterrupt:
            report["run_status"] = "paused_user"
            save(report, CONTEXTUAL_OUTPUT)
            print("Interrupted; completed cases saved.", flush=True)
            return
    report["run_status"] = "completed"
    save(report, CONTEXTUAL_OUTPUT)
    print("Smoke run completed. Automated judgments require review.", flush=True)


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate saved RAG answers with DeepEval."
    )

    parser.add_argument(
        "--mode",
        choices=["baseline", "calibration", "contextual-relevance-smoke"],
        required=True,
        help="Explicitly select the evaluation workflow.",
    )

    parser.add_argument("--preview", action="store_true",
                        help="Inspect contextual smoke inputs without API calls.")
    args = parser.parse_args()
    if args.preview and args.mode != "contextual-relevance-smoke":
        parser.error("--preview is supported only for contextual-relevance-smoke")
    if args.mode == "contextual-relevance-smoke":
        run_contextual_smoke(preview=args.preview)
        return

    if args.mode == "calibration":
        run_calibration()
    else:
        run_baseline()


if __name__ == "__main__":
    main()

