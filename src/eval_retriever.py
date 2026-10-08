"""Evaluate the existing retriever against the source-relative pilot goldens."""

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


GOLDENS_PATH = Path("data/eval/retrieval_goldens.json")
OUTPUT_PATH = Path("data/eval/retrieval_baseline.json")
TOP_K = 5


def corpus_hash(chunks):
    payload = json.dumps(
        chunks, sort_keys=True, ensure_ascii=False, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def score_results(retrieved_ids, relevant_ids):
    ranked = retrieved_ids[:TOP_K]
    if len(ranked) != len(set(ranked)):
        raise ValueError("Duplicate retrieved IDs would distort rank scores")
    relevant = set(relevant_ids)
    if not relevant:
        raise ValueError("This pilot requires answerable cases with relevant IDs")
    first_rank = next(
        (rank for rank, cid in enumerate(ranked, 1) if cid in relevant), None
    )
    return {
        "first_relevant_rank_at_5": first_rank,
        "hit_at_1": int(first_rank is not None and first_rank <= 1),
        "hit_at_3": int(first_rank is not None and first_rank <= 3),
        "hit_at_5": int(first_rank is not None),
        "reciprocal_rank_at_5": 1 / first_rank if first_rank else 0.0,
    }


def self_test():
    fixtures = [
        (["gold", "a", "b", "c", "d"], ["gold"], 1),
        (["a", "gold", "b", "c", "d"], ["gold"], 2),
        (["a", "b", "c", "d", "gold"], ["gold"], 5),
        (["a", "b", "c", "d", "e"], ["gold"], None),
        (["a", "b", "c", "d", "e", "gold"], ["gold"], None),
        (["a", "gold2", "b", "gold1", "c"], ["gold1", "gold2"], 2),
    ]
    for retrieved, relevant, rank in fixtures:
        expected = {
            "first_relevant_rank_at_5": rank,
            "hit_at_1": int(rank is not None and rank <= 1),
            "hit_at_3": int(rank is not None and rank <= 3),
            "hit_at_5": int(rank is not None),
            "reciprocal_rank_at_5": 1 / rank if rank else 0.0,
        }
        if score_results(retrieved, relevant) != expected:
            raise AssertionError(f"Scoring failed for {retrieved}")
    for retrieved, relevant in [(["a", "a"], ["a"]), (["a"], [])]:
        try:
            score_results(retrieved, relevant)
        except ValueError:
            continue
        raise AssertionError("Invalid input was accepted")
    print("PASS: 6 ranking cases and 2 invalid-input checks")


def load_and_validate():
    golden_bytes = GOLDENS_PATH.read_bytes()
    goldens = json.loads(golden_bytes)
    chunks = {}
    for path in sorted(Path("data/chunks").glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        for chunk in data["chunks"]:
            cid = f"{data['video_id']}_{chunk['chunk_id']}"
            if cid in chunks:
                raise ValueError(f"Duplicate source ID: {cid}")
            chunks[cid] = chunk["text"]
    if len(chunks) != goldens["corpus_chunk_count"]:
        raise ValueError("Corpus count differs from golden snapshot")
    if corpus_hash(chunks) != goldens["corpus_text_sha256"]:
        raise ValueError("Corpus text differs from golden snapshot")
    cases = goldens["cases"]
    if not cases or len({case["id"] for case in cases}) != len(cases):
        raise ValueError("Cases must be nonempty and have unique IDs")
    for case in cases:
        ids = case["relevant_chunk_ids"]
        if not case["query"].strip() or not ids or len(ids) != len(set(ids)):
            raise ValueError(f"Invalid case: {case['id']}")
        if any(cid not in chunks for cid in ids):
            raise ValueError(f"Unknown label in case: {case['id']}")
        if {item["chunk_id"] for item in case["evidence"]} != set(ids):
            raise ValueError(f"Evidence IDs differ from labels: {case['id']}")
        for item in case["evidence"]:
            if not item["text_excerpt"] or item["text_excerpt"] not in chunks[item["chunk_id"]]:
                raise ValueError(f"Evidence text mismatch: {case['id']}")
    return goldens, chunks, hashlib.sha256(golden_bytes).hexdigest()


def evaluate():
    # Preserve earlier baseline reports rather than silently replacing them.
    if OUTPUT_PATH.exists():
        raise FileExistsError(f"Report already exists: {OUTPUT_PATH}")
    goldens, chunks, golden_hash = load_and_validate()
    if not Path("data/chroma_db/chroma.sqlite3").is_file():
        raise FileNotFoundError("Existing Chroma database not found")

    from retriever import collection, retrieve

    stored = collection.get(include=["documents"])
    actual = dict(zip(stored["ids"], stored["documents"]))
    if actual != chunks:
        raise ValueError("Index IDs/text differ from current chunks; run check_index.py")

    print(f"Dataset status: {goldens['status']}")
    rows = []
    for case in goldens["cases"]:
        result = retrieve(case["query"], top_k=TOP_K)
        ids = result["ids"][0]
        if len(ids) != TOP_K or any(cid not in chunks for cid in ids):
            raise ValueError(f"Unexpected retrieval output for {case['id']}")
        scores = score_results(ids, case["relevant_chunk_ids"])
        rows.append({
            "id": case["id"], "topic": case["topic"], "query": case["query"],
            "relevant_chunk_ids": case["relevant_chunk_ids"],
            "retrieved_ids": ids, **scores,
        })
        print(f"{case['id']}: first relevant rank = {scores['first_relevant_rank_at_5']}")

    summary = {
        name: sum(row[key] for row in rows) / len(rows)
        for name, key in [
            ("Hit@1", "hit_at_1"), ("Hit@3", "hit_at_3"),
            ("Hit@5", "hit_at_5"), ("MRR@5", "reciprocal_rank_at_5"),
        ]
    }
    packages = {}
    for name in ("chromadb", "sentence-transformers"):
        try:
            packages[name] = version(name)
        except PackageNotFoundError:
            packages[name] = "unknown"
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False
    )
    report = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_name": goldens["dataset_name"], "dataset_status": goldens["status"],
        "goldens_sha256": golden_hash, "corpus_text_sha256": corpus_hash(chunks),
        "git_head": revision.stdout.strip() if revision.returncode == 0 else None,
        "retriever_source_sha256": hashlib.sha256(Path("src/retriever.py").read_bytes()).hexdigest(),
        "evaluator_source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "package_versions": packages, "top_k": TOP_K, "question_count": len(rows),
        "summary": summary, "cases": rows,
    }
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    print("\nPILOT RETRIEVAL RESULTS")
    for metric, value in summary.items():
        print(f"{metric}: {value:.4f}")
    print(f"Saved: {OUTPUT_PATH}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
    else:
        evaluate()
