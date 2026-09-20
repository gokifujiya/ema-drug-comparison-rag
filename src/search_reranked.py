from pathlib import Path
import json
import sys

import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder


EMBEDDINGS_DIR = Path("data/embeddings")

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

CANDIDATE_K = 20
FINAL_K_PER_DRUG = 3


def load_data():
    embeddings = np.load(
        EMBEDDINGS_DIR / "embeddings.npy"
    )

    with (
        EMBEDDINGS_DIR / "chunks.json"
    ).open("r", encoding="utf-8") as f:
        chunks = json.load(f)

    return embeddings, chunks


def retrieve_candidates(
    query,
    model,
    embeddings,
    chunks,
    top_k=CANDIDATE_K,
):
    query_embedding = model.encode(
        query,
        normalize_embeddings=True,
    )

    scores = embeddings @ query_embedding
    indices = np.argsort(scores)[::-1][:top_k]

    candidates = []

    for index in indices:
        candidates.append(
            {
                "bi_score": float(scores[index]),
                "chunk": chunks[index],
            }
        )

    return candidates


def retrieve_candidates(
    query,
    model,
    embeddings,
    chunks,
    top_k=CANDIDATE_K,
):
    query_embedding = model.encode(
        query,
        normalize_embeddings=True,
    )

    scores = embeddings @ query_embedding
    indices = np.argsort(scores)[::-1][:top_k]

    candidates = []

    for index in indices:
        candidates.append(
            {
                "bi_score": float(scores[index]),
                "chunk": chunks[index],
            }
        )

    return candidates


# ADD THIS FUNCTION HERE
def build_reranker_text(chunk: dict) -> str:
    return (
        f"Drug: {chunk['drug']}\n"
        f"Active substance: {chunk['active_substance']}\n"
        f"SmPC section {chunk['section']}: "
        f"{chunk['section_title']}\n\n"
        f"{chunk['text']}"
    )


def rerank(query, candidates, reranker):
    pairs = [
        [
            query,
            build_reranker_text(candidate["chunk"]),
        ]
        for candidate in candidates
    ]

    scores = reranker.predict(pairs)

    for candidate, score in zip(candidates, scores):
        candidate["cross_score"] = float(score)

    return sorted(
        candidates,
        key = lambda x: x["cross_score"],
        reverse = True,
    )


def select_per_drug(
    candidates,
    per_drug=FINAL_K_PER_DRUG,
):
    selected = []

    for drug in ["Jardiance", "Forxiga"]:
        drug_results = [
            item
            for item in candidates
            if item["chunk"]["drug"] == drug
        ]

        selected.extend(drug_results[:per_drug])

    return selected


def print_results(results):
    for rank, result in enumerate(results, start=1):
        chunk = result["chunk"]

        print("\n" + "=" * 80)
        print(
            f"Result {rank} | "
            f"bi={result['bi_score']:.4f} | "
            f"cross={result['cross_score']:.4f}"
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
            'Usage: python src/search_reranked.py '
            '"your question"'
        )
        raise SystemExit(1)

    query = " ".join(sys.argv[1:])

    embeddings, chunks = load_data()

    print(f"Query: {query}")

    print(f"\nLoading embedding model: {EMBEDDING_MODEL}")
    embedding_model = SentenceTransformer(
        EMBEDDING_MODEL
    )

    print(f"Loading reranker: {RERANKER_MODEL}")
    reranker = CrossEncoder(
        RERANKER_MODEL
    )

    candidates = retrieve_candidates(
        query,
        embedding_model,
        embeddings,
        chunks,
    )

    reranked = rerank(
        query,
        candidates,
        reranker,
    )

    results = select_per_drug(
        reranked,
        per_drug = FINAL_K_PER_DRUG,
    )

    print_results(results)


if __name__ == "__main__":
    main()
