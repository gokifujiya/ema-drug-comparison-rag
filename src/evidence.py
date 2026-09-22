from src.retriever import retrieve, load_chunks


def format_pages(pages: list[int]) -> str:
    if not pages:
        return "unknown"

    if len(pages) == 1:
        return str(pages[0])

    return f"{pages[0]}-{pages[-1]}"


def get_neighbor_chunks(item: dict, all_chunks: list[dict]) -> list[dict]:
    """
    Return immediately adjacent chunks from the same drug and SmPC section.
    """

    current_chunk = next(
        (
            chunk
            for chunk in all_chunks
            if chunk["chunk_id"] == item["chunk_id"]
        ),
        None,
    )

    if current_chunk is None:
        return []

    current_index = current_chunk["chunk_index"]

    same_section = [
        chunk
        for chunk in all_chunks
        if chunk["drug"] == item["drug"]
        and chunk["section"] == item["section"]
    ]

    same_section.sort(key=lambda chunk: chunk["chunk_index"])

    return [
        chunk
        for chunk in same_section
        if abs(chunk["chunk_index"] - current_index) == 1
    ]


def build_evidence(query: str) -> dict:
    retrieval = retrieve(query)
    all_chunks = load_chunks()

    evidence_by_drug = {}

    for drug, items in retrieval["evidence"].items():
        expanded_items = []
        seen_chunk_ids = set()

        for item in items:
            candidates = [item]

            candidates.extend(
                get_neighbor_chunks(item, all_chunks)
            )

            for candidate in candidates:
                if candidate["chunk_id"] in seen_chunk_ids:
                    continue

                seen_chunk_ids.add(candidate["chunk_id"])
                expanded_items.append(candidate)

        evidence_by_drug[drug] = []

        for rank, item in enumerate(expanded_items, start=1):
            evidence_by_drug[drug].append(
                {
                    "evidence_id": f"{drug.lower()}_{rank}",
                    "drug": item["drug"],
                    "active_substance": item["active_substance"],
                    "section": item["section"],
                    "section_title": item["section_title"],
                    "pages": item["pages"],
                    "page_label": format_pages(item["pages"]),
                    "chunk_id": item["chunk_id"],
                    "text": item["text"],
                    "bi_score": item.get("bi_score"),
                    "cross_score": item.get("cross_score"),
                }
            )

    return {
        "query": query,
        "evidence": evidence_by_drug,
    }


def build_context(evidence_package: dict) -> str:
    blocks = []

    for drug, items in evidence_package["evidence"].items():
        for item in items:
            block = (
                f"[{item['evidence_id']}]\n"
                f"Drug: {item['drug']} "
                f"({item['active_substance']})\n"
                f"SmPC section {item['section']}: "
                f"{item['section_title']}\n"
                f"Pages: {item['page_label']}\n"
                f"Chunk ID: {item['chunk_id']}\n\n"
                f"{item['text']}"
            )

            blocks.append(block)

    return "\n\n---\n\n".join(blocks)
