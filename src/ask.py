from src.generator import generate_answer


def format_pages(pages: list[int]) -> str:
    if not pages:
        return "unknown"

    if len(pages) == 1:
        return str(pages[0])

    return f"{pages[0]}–{pages[-1]}"


def print_sources(evidence: dict) -> None:
    print("\nSources")
    print("=" * 70)

    for drug, items in evidence.items():
        for index, item in enumerate(items, start=1):
            evidence_id = f"{drug.lower()}_{index}"

            print(f"\n[{evidence_id}]")
            print(
                f"{item['drug']} "
                f"({item['active_substance']})"
            )
            print("EMA Summary of Product Characteristics (SmPC)")
            print(
                f"Section {item['section']} — "
                f"{item['section_title']}"
            )
            print(f"Page(s): {format_pages(item['pages'])}")
            print(f"Chunk ID: {item['chunk_id']}")


def main() -> None:
    print("EMA Drug Comparison RAG")
    print("Jardiance (empagliflozin) vs Forxiga (dapagliflozin)")
    print("=" * 70)

    query = input("\nAsk a question: ").strip()

    if not query:
        print("No question entered.")
        return

    print("\nRetrieving EMA evidence and generating answer...\n")

    result = generate_answer(query)

    print("Answer")
    print("=" * 70)
    print(result["answer"])

    print_sources(result["evidence"])


if __name__ == "__main__":
    main()
