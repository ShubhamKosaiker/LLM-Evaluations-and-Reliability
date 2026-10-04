from retriever import retrieve
from generator import generate_answer


def run_rag(query, top_k=3):
    results = retrieve(query, top_k=top_k)

    contexts = results["documents"][0]
    metadatas = results["metadatas"][0]

    answer = generate_answer(query, contexts)

    return {
        "query": query,
        "answer": answer,
        "contexts": contexts,
        "sources": metadatas,
    }


if __name__ == "__main__":
    query = "What are the red flags in a patient with cough?"

    result = run_rag(query)

    print("\nANSWER:\n")
    print(result["answer"])

    print("\nSOURCES:\n")
    for source in result["sources"]:
        print(
            f"{source['title']} | "
            f"{source['start']:.1f}s - {source['end']:.1f}s"
        )