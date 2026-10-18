
"""DeepEval faithfulness evaluation for saved RAG answers.

Modes:
    baseline     - Existing pilot and validation evaluations.
    calibration  - Five controlled evaluator-reliability cases.

Does not rerun retrieval or answer generation.
"""

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from deepeval.metrics import FaithfulnessMetric
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


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate saved RAG answers with DeepEval."
    )

    parser.add_argument(
        "--mode",
        choices=["baseline", "calibration"],
        required=True,
        help="Explicitly select the evaluation workflow.",
    )

    args = parser.parse_args()

    if args.mode == "calibration":
        run_calibration()
    else:
        run_baseline()


if __name__ == "__main__":
    main()
