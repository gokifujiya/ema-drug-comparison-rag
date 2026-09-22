from pathlib import Path
import json

import numpy as np
from sentence_transformers import SentenceTransformer, CrossEncoder


EMBEDDINGS_DIR = Path("data/embeddings")

BI_ENCODER_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
CROSS_ENCODER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

CANDIDATE_K = 20
FINAL_K_PER_DRUG = 3
_bi_encoder = None
_cross_encoder = None


def get_bi_encoder() -> SentenceTransformer:
    global _bi_encoder

    if _bi_encoder is None:
        print(f"Loading embedding model: {BI_ENCODER_MODEL}")
        _bi_encoder = SentenceTransformer(BI_ENCODER_MODEL)

    return _bi_encoder


def get_cross_encoder() -> CrossEncoder:
    global _cross_encoder

    if _cross_encoder is None:
        print(f"Loading reranker: {CROSS_ENCODER_MODEL}")
        _cross_encoder = CrossEncoder(CROSS_ENCODER_MODEL)

    return _cross_encoder


def load_chunks() -> list[dict]:
    path = EMBEDDINGS_DIR / "chunks.json"

    with path.open("r", encoding = "utf-8") as f:
        return json.load(f)


def load_embeddings() -> np.ndarray:
    return np.load(EMBEDDINGS_DIR / "embeddings.npy")


def build_embedding_text(chunk: dict) -> str:
    return (
        f"Drug: {chunk['drug']}\n"
        f"Active substance: {chunk['active_substance']}\n"
        f"SmPC section {chunk['section']}: "
        f"{chunk['section_title']}\n\n"
        f"{chunk['text']}"
    )


def build_reranker_text(chunk: dict) -> str:
    return build_embedding_text(chunk)


def retrieve_candidates(
    query: str,
    chunks: list[dict],
    embeddings: np.ndarray,
    model: SentenceTransformer,
    candidate_k: int = CANDIDATE_K,
) -> list[dict]:

    query_embedding = model.encode(
        query,
        normalize_embeddings = True,
    )

    scores = embeddings @ query_embedding

    top_indices = np.argsort(scores)[::-1][:candidate_k]

    candidates = []

    for index in top_indices:
        candidates.append(
            {
                "chunk": chunks[index],
                "bi_score": float(scores[index]),
            }
        )

    return candidates


def retrieve_candidates_per_drug(
    query: str,
    chunks: list[dict],
    embeddings: np.ndarray,
    model: SentenceTransformer,
    candidate_k_per_drug: int = CANDIDATE_K,
) -> list[dict]:

    query_embedding = model.encode(
        query,
        normalize_embeddings = True,
    )

    scores = embeddings @ query_embedding

    candidates = []

    drugs = sorted(set(chunk["drug"] for chunk in chunks))

    for drug in drugs:
        drug_indices = [
            index
            for index, chunk in enumerate(chunks)
            if chunk["drug"] == drug
        ]

        ranked_indices = sorted(
            drug_indices,
            key = lambda index: scores[index],
            reverse = True,
        )

        top_indices = ranked_indices[:candidate_k_per_drug]

        for index in top_indices:
            candidates.append(
                {
                    "chunk": chunks[index],
                    "bi_score": float(scores[index]),
                }
            )

    return candidates


def rerank(
    query: str,
    candidates: list[dict],
    model: CrossEncoder,
) -> list[dict]:

    pairs = [
        [
            query,
            build_reranker_text(candidate["chunk"]),
        ]
        for candidate in candidates
    ]

    cross_scores = model.predict(pairs)

    results = []

    for candidate, score in zip(candidates, cross_scores):
        results.append(
            {
                **candidate,
                "cross_score": float(score),
            }
        )

    results.sort(
        key = lambda item: item["cross_score"],
        reverse = True,
    )

    return results


def select_per_drug(
    results: list[dict],
    k_per_drug: int = FINAL_K_PER_DRUG,
) -> dict[str, list[dict]]:

    selected = {}

    for result in results:
        drug = result["chunk"]["drug"]

        if drug not in selected:
            selected[drug] = []

        if len(selected[drug]) < k_per_drug:
            selected[drug].append(result)

    return selected


def retrieve(
    query: str,
    candidate_k: int = CANDIDATE_K,
    k_per_drug: int = FINAL_K_PER_DRUG,
) -> dict:

    chunks = load_chunks()
    embeddings = load_embeddings()

    bi_encoder = get_bi_encoder()
    cross_encoder = get_cross_encoder() 

    candidates = retrieve_candidates_per_drug(
        query = query,
        chunks = chunks,
        embeddings = embeddings,
        model = bi_encoder,
        candidate_k_per_drug = candidate_k,
    )

    reranked = rerank(
        query = query,
        candidates = candidates,
        model = cross_encoder,
    )

    selected = select_per_drug(
        reranked,
        k_per_drug = k_per_drug,
    )

    evidence = {}

    for drug, results in selected.items():
        evidence[drug] = []

        for result in results:
            chunk = result["chunk"]

            evidence[drug].append(
                {
                    "chunk_id": chunk["chunk_id"],
                    "drug": chunk["drug"],
                    "active_substance": chunk["active_substance"],
                    "section": chunk["section"],
                    "section_title": chunk["section_title"],
                    "pages": chunk["pages"],
                    "text": chunk["text"],
                    "bi_score": result["bi_score"],
                    "cross_score": result["cross_score"],
                }
            )

    return {
        "query": query,
        "evidence": evidence,
    }
