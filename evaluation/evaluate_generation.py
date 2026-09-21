import json
import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from src.generator import generate_answer


load_dotenv()

QUESTIONS_PATH = Path("evaluation/questions.json")
RESULTS_PATH = Path("evaluation/generation_results.json")

JUDGE_MODEL = "gpt-5.6"


JUDGE_INSTRUCTIONS = """
You are evaluating a pharmaceutical RAG system.

You will receive:
1. A question.
2. A generated answer.
3. A list of expected factual points.

Evaluate only whether the generated answer covers the expected points
accurately and without unsupported conclusions.

Important rules:
- Do not require exact wording.
- Accept semantically equivalent statements.
- Do not penalize concise answers merely for being concise.
- Do not require a specific retrieved chunk or exact passage.
- An expected point counts as covered if its clinically meaningful content
  is present in the answer.
- Do not infer that one medicine is safer or more effective unless the
  generated answer has evidence for that conclusion.
- Preserve distinctions in regulatory wording such as "not recommended",
  "should be considered", and "recommended".
- A statement that contradicts an expected point is an error.
- Unsupported comparative conclusions are errors.

Return valid JSON only, using this structure:

{
  "covered_points": [0, 1],
  "missing_points": [2],
  "contradictions": [],
  "unsupported_claims": [],
  "summary": "Brief explanation"
}

The point numbers are zero-based indexes into the expected_points list.
"""


def load_questions():
    with open(QUESTIONS_PATH, "r", encoding = "utf-8") as f:
        return json.load(f)


def get_client():
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set.")

    return OpenAI(api_key = api_key)


def judge_answer(client, item, answer):
    payload = {
        "question": item["question"],
        "generated_answer": answer,
        "expected_points": item["expected_points"],
    }

    response = client.responses.create(
        model = JUDGE_MODEL,
        instructions = JUDGE_INSTRUCTIONS,
        input = json.dumps(payload, ensure_ascii = False, indent = 2),
    )

    text = response.output_text.strip()

    # Remove Markdown fences if the model unexpectedly returns them.
    if text.startswith("```"):
        lines = text.splitlines()
        lines = [
            line
            for line in lines
            if not line.strip().startswith("```")
        ]
        text = "\n".join(lines).strip()

    return json.loads(text)


def evaluate_item(client, item):
    generation = generate_answer(item["question"])
    answer = generation["answer"]

    judgment = judge_answer(
        client = client,
        item = item,
        answer = answer,
    )

    total_points = len(item["expected_points"])
    covered = len(set(judgment["covered_points"]))

    coverage = (
        covered / total_points
        if total_points
        else 0.0
    )

    return {
        "id": item["id"],
        "section": item["section"],
        "topic": item["topic"],
        "question": item["question"],
        "answer": answer,
        "expected_points": item["expected_points"],
        "coverage": coverage,
        "judgment": judgment,
    }


def main():
    questions = load_questions()
    client = get_client()

    results = []

    print("\nEMA RAG Generation Evaluation")
    print("=" * 60)

    for index, item in enumerate(questions, start=1):
        print(
            f"[{index}/{len(questions)}] "
            f"{item['id']}: {item['topic']}"
        )

        result = evaluate_item(client, item)
        results.append(result)

        print(
            f"  coverage: {result['coverage']:.1%}"
        )

        if result["judgment"]["contradictions"]:
            print(
                "  contradictions:",
                result["judgment"]["contradictions"],
            )

        if result["judgment"]["unsupported_claims"]:
            print(
                "  unsupported claims:",
                result["judgment"]["unsupported_claims"],
            )

    average_coverage = (
        sum(result["coverage"] for result in results)
        / len(results)
        if results
        else 0.0
    )

    perfect_answers = sum(
        result["coverage"] == 1.0
        and not result["judgment"]["contradictions"]
        and not result["judgment"]["unsupported_claims"]
        for result in results
    )

    output = {
        "summary": {
            "questions": len(results),
            "average_expected_point_coverage": average_coverage,
            "perfect_answers": perfect_answers,
        },
        "results": results,
    }

    RESULTS_PATH.parent.mkdir(parents = True, exist_ok = True)

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(
            output,
            f,
            ensure_ascii = False,
            indent = 2,
        )

    print("\nSummary")
    print("=" * 60)
    print(f"Questions: {len(results)}")
    print(
        "Average expected-point coverage: "
        f"{average_coverage:.1%}"
    )
    print(
        f"Perfect answers: "
        f"{perfect_answers}/{len(results)}"
    )
    print(f"\nResults saved to: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
