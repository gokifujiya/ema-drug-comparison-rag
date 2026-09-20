from pathlib import Path
import json
import sys

import numpy as np
from sentence_transformers import SentenceTransformer


EMBEDDINGS_DIR = Path("data/embeddings")

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def load_data():
    embeddings = np.load(
        EMBEDDINGS_DIR / "embeddings.npy"
    )

    with (
        EMBEDDINGS_DIR / "chunks.json"
    ).open("r", encoding="utf-8") as f:
        chunks = json.load(f)

    return embeddings, chunks


def search(
    query: str,
    model: SentenceTransformer,
    embeddings: np.ndarray,
    chunks: list[dict],
    top_k: int = 8,
):
    query_embedding = model.encode(
        query,
        normalize_embeddings=True,
    )

    # Embeddings are normalized, so dot product = cosine similarity.
    scores = embeddings @ query_embedding

    top_indices = np.argsort(scores)[::-1][:top_k]

    results = []

    for index in top_indices:
        chunk = chunks[index]

        results.append(
            {
                "score": float(scores[index]),
                "chunk": chunk,
            }
        )

    return results


def print_results(results):
    for rank, result in enumerate(results, start=1):
        chunk = result["chunk"]

        print("\n" + "=" * 80)
        print(
            f"Rank {rank} | "
            f"score={result['score']:.4f}"
        )
        print(
            f"{chunk['drug']} "
            f"({chunk['active_substance']})"
        )
        print(
            f"SmPC {chunk['section']} — "
            f"{chunk['section_title']}"
        )
        print(f"Pages: {chunk['pages']}")
        print(f"Chunk: {chunk['chunk_id']}")
        print("-" * 80)

        text = chunk["text"]

        if len(text) > 1200:
            text = text[:1200] + "\n[...]"

        print(text)


def main():
    if len(sys.argv) < 2:
        print(
            'Usage: python src/search.py '
            '"your question"'
        )
        raise SystemExit(1)

    query = " ".join(sys.argv[1:])

    embeddings, chunks = load_data()

    print(f"Loading model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)

    print(f"\nQuery: {query}")

    results = search(
        query = query,
        model = model,
        embeddings = embeddings,
        chunks = chunks,
        top_k = 8,
    )

    print_results(results)


if __name__ == "__main__":
    main()
