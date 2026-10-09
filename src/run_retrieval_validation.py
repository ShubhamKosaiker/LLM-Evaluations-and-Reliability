import argparse
from pathlib import Path

import eval_retriever as evaluator


parser = argparse.ArgumentParser()
parser.add_argument("mode", choices=["baseline", "topic"])
args = parser.parse_args()

evaluator.GOLDENS_PATH = Path(
    "data/eval/retrieval_validation_goldens.json"
)

baseline_path = Path("data/eval/retrieval_validation_baseline.json")

if args.mode == "baseline":
    evaluator.OUTPUT_PATH = baseline_path
    evaluator.evaluate()
else:
    import experiment_query_topic as experiment

    experiment.GOLDENS_PATH = evaluator.GOLDENS_PATH
    experiment.BASELINE_PATH = baseline_path
    experiment.OUTPUT_PATH = Path(
        "data/eval/retrieval_validation_topic.json"
    )
    experiment.main()