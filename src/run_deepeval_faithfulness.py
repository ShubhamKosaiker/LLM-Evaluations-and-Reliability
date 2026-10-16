
"""Evaluate saved RAG answers with DeepEval faithfulness.

Does not rerun retrieval or answer generation.
Saves progress after every case for resuming.
"""

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


def save(report):
    report["updated_at_utc"] = datetime.now(timezone.utc).isoformat()
    temp = OUTPUT.with_suffix(".tmp")
    temp.write_text(json.dumps(report, indent=2), encoding="utf-8")
    temp.replace(OUTPUT)


def main():
    if OUTPUT.exists():
        report = json.loads(OUTPUT.read_text(encoding="utf-8"))
        if report.get("sources") != SOURCES:
            raise ValueError("Existing report uses different input sources")
        print("Resuming existing report", flush=True)
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
        data = json.loads(Path(source).read_text(encoding="utf-8-sig"))

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
            print(f"{case_id}: {result['status']}", flush=True)

    print("Run finished. Results saved to:", OUTPUT, flush=True)


if __name__ == "__main__":
    main()
