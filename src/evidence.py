from src.retriever import retrieve


def format_pages(pages: list[int]) -> str:
    if not pages:
        return "unknown"

    if len(pages) == 1:
        return str(pages[0])

    return f"{pages[0]}–{pages[-1]}"


def build_evidence(query: str) -> dict:
    retrieval = retrieve(query)

    evidence_by_drug = {}

    for drug, items in retrieval["evidence"].items():
        evidence_by_drug[drug] = []

        for rank, item in enumerate(items, start = 1):
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
                    "bi_score": item["bi_score"],
                    "cross_score": item["cross_score"],
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
