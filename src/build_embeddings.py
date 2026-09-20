from pathlib import Path
import json

import numpy as np
from sentence_transformers import SentenceTransformer


PROCESSED_DIR = Path("data/processed")
EMBEDDINGS_DIR = Path("data/embeddings")

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def load_chunks() -> list[dict]:
    all_chunks = []

    for path in sorted(PROCESSED_DIR.glob("*_chunks.json")):
        with path.open("r", encoding = "utf-8") as f:
            data = json.load(f)

        all_chunks.extend(data["chunks"])

    return all_chunks


def build_embedding_text(chunk: dict) -> str:
    """
    Add regulatory context to the text being embedded.
    """
    return (
        f"Drug: {chunk['drug']}\n"
        f"Active substance: {chunk['active_substance']}\n"
        f"SmPC section {chunk['section']}: "
        f"{chunk['section_title']}\n\n"
        f"{chunk['text']}"
    )


def main() -> None:
    EMBEDDINGS_DIR.mkdir(parents = True, exist_ok = True)

    chunks = load_chunks()

    print(f"Loaded {len(chunks)} chunks")
    print(f"Loading embedding model: {MODEL_NAME}")

    model = SentenceTransformer(MODEL_NAME)

    texts = [
        build_embedding_text(chunk)
        for chunk in chunks
    ]

    print("Creating embeddings...")

    embeddings = model.encode(
        texts,
        normalize_embeddings = True,
        show_progress_bar = True,
    )

    embeddings = np.asarray(
        embeddings,
        dtype = np.float32,
    )

    np.save(
        EMBEDDINGS_DIR / "embeddings.npy",
        embeddings,
    )

    with (
        EMBEDDINGS_DIR / "chunks.json"
    ).open("w", encoding = "utf-8") as f:
        json.dump(
            chunks,
            f,
            ensure_ascii = False,
            indent = 2,
        )

    metadata = {
        "model": MODEL_NAME,
        "number_of_chunks": len(chunks),
        "embedding_dimension": int(embeddings.shape[1]),
    }

    with (
        EMBEDDINGS_DIR / "metadata.json"
    ).open("w", encoding = "utf-8") as f:
        json.dump(
            metadata,
            f,
            ensure_ascii = False,
            indent = 2,
        )

    print()
    print(f"Embeddings shape: {embeddings.shape}")
    print(
        "Saved embeddings -> "
        "data/embeddings/embeddings.npy"
    )
    print(
        "Saved chunks -> "
        "data/embeddings/chunks.json"
    )
    print(
        "Saved metadata -> "
        "data/embeddings/metadata.json"
    )


if __name__ == "__main__":
    main()
