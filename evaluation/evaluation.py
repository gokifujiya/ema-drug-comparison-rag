import json
from pathlib import Path

from src.retriever import retrieve


QUESTIONS_PATH = Path("evaluation/questions.json")


def load_questions():
    with open(QUESTIONS_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def evaluate_retrieval(item):
    result = retrieve(item["question"])

    expected_sections = item["reference_sections"]
    evidence = result["evidence"]

    drug_results = {}

    for drug in ["Jardiance", "Forxiga"]:
        expected_section = expected_sections[drug]

        retrieved_sections = [
            chunk["section"]
            for chunk in evidence.get(drug, [])
        ]

        hit = expected_section in retrieved_sections

        drug_results[drug] = {
            "expected_section": expected_section,
            "retrieved_sections": retrieved_sections,
            "hit": hit,
        }

    both_hit = all(
        result["hit"]
        for result in drug_results.values()
    )

    return {
        "id": item["id"],
        "question": item["question"],
        "drug_results": drug_results,
        "both_hit": both_hit,
    }


def main():
    questions = load_questions()

    results = [
        evaluate_retrieval(item)
        for item in questions
    ]

    jardiance_hits = sum(
        r["drug_results"]["Jardiance"]["hit"]
        for r in results
    )

    forxiga_hits = sum(
        r["drug_results"]["Forxiga"]["hit"]
        for r in results
    )

    both_hits = sum(
        r["both_hit"]
        for r in results
    )

    total = len(results)

    print("\nEMA RAG Retrieval Evaluation")
    print("=" * 60)

    for result in results:
        status = "PASS" if result["both_hit"] else "FAIL"

        print(
            f"{result['id']}: {status}"
        )

        for drug in ["Jardiance", "Forxiga"]:
            info = result["drug_results"][drug]

            print(
                f"  {drug}: "
                f"expected={info['expected_section']} "
                f"retrieved={info['retrieved_sections']} "
                f"hit={info['hit']}"
            )

    print("\nSummary")
    print("=" * 60)
    print(
        f"Jardiance section hit rate: "
        f"{jardiance_hits}/{total} "
        f"({jardiance_hits / total:.1%})"
    )
    print(
        f"Forxiga section hit rate: "
        f"{forxiga_hits}/{total} "
        f"({forxiga_hits / total:.1%})"
    )
    print(
        f"Both-drug section hit rate: "
        f"{both_hits}/{total} "
        f"({both_hits / total:.1%})"
    )


if __name__ == "__main__":
    main()
