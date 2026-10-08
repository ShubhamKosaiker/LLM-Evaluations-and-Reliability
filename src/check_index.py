import json
from pathlib import Path

import chromadb


CHUNKS_DIR = Path("data/chunks")
DB_DIR = Path("data/chroma_db")
COLLECTION_NAME = "medical_videos"


def main():
    expected = {}

    for path in sorted(CHUNKS_DIR.glob("*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))

        for chunk in data["chunks"]:
            record_id = f"{data['video_id']}_{chunk['chunk_id']}"

            if record_id in expected:
                raise ValueError(f"Duplicate chunk ID: {record_id}")

            expected[record_id] = chunk["text"]

    if not expected:
        raise ValueError("No chunks found in data/chunks")

    if not DB_DIR.is_dir():
        raise FileNotFoundError(f"Database directory missing: {DB_DIR}")

    client = chromadb.PersistentClient(path=str(DB_DIR))
    collection = client.get_collection(COLLECTION_NAME)
    stored = collection.get(include=["documents"])

    actual = dict(zip(stored["ids"], stored["documents"]))

    missing = sorted(expected.keys() - actual.keys())
    extra = sorted(actual.keys() - expected.keys())
    mismatched = sorted(
        record_id
        for record_id in expected.keys() & actual.keys()
        if expected[record_id] != actual[record_id]
    )

    print(f"Expected chunks: {len(expected)}")
    print(f"Stored records: {len(actual)}")

    for label, failures in [
        ("Missing IDs", missing),
        ("Extra IDs", extra),
        ("Text mismatches", mismatched),
    ]:
        print(f"{label}: {len(failures)}")
        for record_id in failures[:10]:
            print(f"  {record_id}")

    if missing or extra or mismatched:
        raise SystemExit("FAIL: index does not match current chunks")

    print("PASS: index IDs and text match current chunks")


if __name__ == "__main__":
    main()