import os

from dotenv import load_dotenv
from openai import OpenAI

from src.evidence import build_evidence, build_context


load_dotenv()

MODEL = "gpt-5.6"


SYSTEM_PROMPT = """
You are a pharmaceutical regulatory information assistant.

Answer the user's question using only the EMA Summary of Product
Characteristics (SmPC) evidence provided in the context.

Rules:
1. Do not use outside medical knowledge.
2. Do not make claims that are not supported by the supplied evidence.
3. Compare the medicines directly when the question asks for a comparison.
4. Preserve clinically important differences between the medicines.
5. Cite factual claims using the evidence IDs exactly as supplied,
   for example [jardiance_1] or [forxiga_1].
6. Do not invent evidence IDs.
7. If the supplied evidence is insufficient to answer part of the question,
   explicitly say that the available evidence is insufficient.
8. Be concise but medically precise.
"""


def get_client() -> OpenAI:
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise RuntimeError(
            "OPENAI_API_KEY environment variable is not set."
        )

    return OpenAI(api_key=api_key)


def build_prompt(
    query: str,
    context: str,
) -> str:
    return f"""
USER QUESTION
{query}

EMA SmPC EVIDENCE
{context}

TASK
Answer the user question using only the evidence above.

Cite each important factual claim with the relevant evidence ID.
When comparing the two medicines, make similarities and differences clear.
"""


def generate_answer(query: str) -> dict:
    evidence_package = build_evidence(query)
    context = build_context(evidence_package)

    client = get_client()

    response = client.responses.create(
        model = MODEL,
        instructions = SYSTEM_PROMPT,
        input = build_prompt(
            query = query,
            context = context,
        ),
    )

    answer = response.output_text

    return {
        "query": query,
        "answer": answer,
        "evidence": evidence_package["evidence"],
    }
