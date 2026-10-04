import json
from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction


CHUNKS_DIR = Path("data/chunks")
DB_DIR = "data/chroma_db"
COLLECTION_NAME = "medical_videos"

embedding_function = SentenceTransformerEmbeddingFunction(
    model_name="all-MiniLM-L6-v2"
)

client = chromadb.PersistentClient(path=DB_DIR)

collection = client.get_or_create_collection(
    name=COLLECTION_NAME,
    embedding_function=embedding_function
)


def index_chunks():
    total = 0

    for file_path in CHUNKS_DIR.glob("*.json"):

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for chunk in data["chunks"]:

            chunk_id = f"{data['video_id']}_{chunk['chunk_id']}"

            collection.upsert(
                ids=[chunk_id],
                documents=[chunk["text"]],
                metadatas=[{
                    "video_id": data["video_id"],
                    "title": data["title"],
                    "url": data["url"],
                    "source": data["source"],
                    "start": chunk["start"],
                    "end": chunk["end"],
                    "chunk_id": chunk["chunk_id"]
                }]
            )

            total += 1

    print(f"Indexed {total} chunks")


def retrieve(query, top_k=3):

    results = collection.query(
        query_texts=[query],
        n_results=top_k
    )

    return results


if __name__ == "__main__":

    index_chunks()

    query = "When should a chest x-ray be ordered for cough?"

    results = retrieve(query)

    for i, text in enumerate(results["documents"][0], 1):

        metadata = results["metadatas"][0][i - 1]

        print(f"\n--- RESULT {i} ---")
        print(f"Video: {metadata['title']}")
        print(f"Time: {metadata['start']:.1f}s - {metadata['end']:.1f}s")
        print(text[:500])