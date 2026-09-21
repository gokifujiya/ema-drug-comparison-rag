from src.retriever import retrieve


FAILURE_QUERIES = {
    "eval_003": (
        "Compare the recommendations for Jardiance and Forxiga in patients "
        "with hepatic impairment, elderly patients, and paediatric patients, "
        "and compare their methods of administration."
    ),
    "eval_006": (
        "Compare the warnings and precautions for volume depletion and "
        "hypotension with Jardiance and Forxiga."
    ),
    "eval_010": (
        "Compare the warnings about lower limb amputations for Jardiance "
        "and Forxiga."
    ),
    "eval_013": (
        "Compare the effects of other medicinal products on the "
        "pharmacokinetics of Jardiance and Forxiga."
    ),
    "eval_014": (
        "Compare the effects of Jardiance and Forxiga on other medicinal "
        "products, including their interactions with lithium and other "
        "coadministered medicines."
    ),
    "eval_019": (
        "Compare the safety profiles of Jardiance and Forxiga in patients "
        "with type 2 diabetes mellitus based on the provided SmPC passages."
    ),
}


def print_chunk(chunk):
    text = " ".join(chunk["text"].split())

    if len(text) > 500:
        text = text[:500] + "..."

    print(f"    chunk_id:    {chunk['chunk_id']}")
    print(f"    section:     {chunk['section']} {chunk['section_title']}")
    print(f"    pages:       {chunk['pages']}")
    print(f"    bi_score:    {chunk['bi_score']:.4f}")
    print(f"    cross_score: {chunk['cross_score']:.4f}")
    print(f"    text:        {text}")
    print()


def main():
    for eval_id, query in FAILURE_QUERIES.items():
        print("\n" + "=" * 80)
        print(eval_id)
        print(query)
        print("=" * 80)

        result = retrieve(query)

        for drug in ["Jardiance", "Forxiga"]:
            print(f"\n{drug}")
            print("-" * 80)

            for chunk in result["evidence"][drug]:
                print_chunk(chunk)


if __name__ == "__main__":
    main()
