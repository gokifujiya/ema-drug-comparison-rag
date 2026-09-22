import json
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from src.evidence import build_evidence, build_context


QUESTIONS_FILE = Path("evaluation/questions.json")
RESULTS_FILE = Path("evaluation/evidence_results.json")

MODEL = "gpt-5.6"

load_dotenv()
client = OpenAI()


def judge_evidence(
    question: str,
    expected_points: list[str],
    context: str,
) -> dict:

    expected_text = "\n".join(
        f"{index + 1}. {point}"
        for index, point in enumerate(expected_points)
    )

    prompt = f"""
You are evaluating evidence retrieved by a regulatory RAG system.

QUESTION:
{question}

EXPECTED POINTS:
{expected_text}

RETRIEVED EMA SmPC EVIDENCE:
{context}

For each expected point, determine whether the retrieved evidence
contains enough information to support that point.

Important rules:
- Evaluate the evidence, not whether an answer was generated.
- Semantic equivalence is sufficient.
- Do not require exact wording.
- A point is supported only if the supplied evidence contains the
  information needed to establish it.
- Do not use outside medical knowledge.
- Do not infer missing facts.
- For comparative points, the evidence must contain enough information
  to support the comparison.

Return ONLY valid JSON in this format:

{{
  "points": [
    {{
      "point": 1,
      "supported": true,
      "reason": "brief reason"
    }}
  ]
}}
"""

    response = client.responses.create(
        model = MODEL,
        input = prompt,
    )

    return json.loads(response.output_text)


def main():
    with open(QUESTIONS_FILE, encoding="utf-8") as f:
        questions = json.load(f)

    results = []
    total_supported = 0
    total_points = 0

    print("\nEMA RAG Evidence Coverage Evaluation")
    print("=" * 60)

    for index, item in enumerate(questions, start=1):
        print(
            f"[{index}/{len(questions)}] "
            f"{item['id']}: {item['topic']}"
        )

        evidence_package = build_evidence(item["question"])
        context = build_context(evidence_package)

        judgment = judge_evidence(
            question = item["question"],
            expected_points = item["expected_points"],
            context = context,
        )

        points = judgment["points"]

        supported = sum(
            1
            for point in points
            if point["supported"]
        )

        expected_count = len(item["expected_points"])

        coverage = (
            supported / expected_count
            if expected_count
            else 0.0
        )

        total_supported += supported
        total_points += expected_count

        print(f"  evidence coverage: {coverage:.1%}")

        unsupported = [
            point
            for point in points
            if not point["supported"]
        ]

        for point in unsupported:
            print(
                f"    missing point {point['point']}: "
                f"{point['reason']}"
            )

        results.append(
            {
                "id": item["id"],
                "topic": item["topic"],
                "supported_points": supported,
                "expected_points": expected_count,
                "coverage": coverage,
                "judgment": judgment,
            }
        )

    overall_coverage = (
        total_supported / total_points
        if total_points
        else 0.0
    )

    output = {
        "questions": len(questions),
        "supported_points": total_supported,
        "expected_points": total_points,
        "overall_evidence_coverage": overall_coverage,
        "results": results,
    }

    with open(
        RESULTS_FILE,
        "w",
        encoding = "utf-8",
    ) as f:
        json.dump(
            output,
            f,
            indent = 2,
            ensure_ascii=False,
        )

    print("\nSummary")
    print("=" * 60)
    print(f"Questions: {len(questions)}")
    print(
        f"Overall evidence coverage: "
        f"{overall_coverage:.1%}"
    )
    print(
        f"Supported expected points: "
        f"{total_supported}/{total_points}"
    )
    print(f"\nResults saved to: {RESULTS_FILE}")


if __name__ == "__main__":
    main()
