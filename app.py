from __future__ import annotations

import html
from pathlib import Path

import gradio as gr

from src.evidence_highlighter import highlight_text
from src.generator import generate_answer

DRUG_ORDER = ("Jardiance", "Forxiga")

DISCLAIMER = (
    "Demonstration system using EMA Summary of Product Characteristics (SmPC) "
    "evidence for Jardiance (empagliflozin) and Forxiga (dapagliflozin). "
    "Not medical advice. Not a substitute for the official product information "
    "or professional clinical judgment. Comparative statements are limited to "
    "the retrieved SmPC passages."
)


def format_evidence_html(evidence: dict | None, answer: str = "") -> str:
    if not evidence:
        return (
            "<p><em>No EMA SmPC evidence for this turn yet. "
            "Ask a comparison question to retrieve sources.</em></p>"
        )

    sections = [
        "<p><strong>Sources: EMA Summary of Product Characteristics (SmPC)</strong></p>"
    ]

    for drug in DRUG_ORDER:
        items = evidence.get(drug, [])
        if not items:
            continue

        sections.append(f"<h3>{html.escape(drug)}</h3>")
        for item in items:
            evidence_id = html.escape(str(item["evidence_id"]))
            pages = html.escape(str(item.get("page_label") or "unknown"))
            section = html.escape(str(item.get("section", "")))
            title = html.escape(str(item.get("section_title", "")))
            chunk_id = html.escape(str(item.get("chunk_id", "")))
            body = highlight_text(item.get("text") or "", answer)

            sections.append(
                f"<article style='margin-bottom:1rem;padding:0.75rem;"
                f"border:1px solid #ddd;border-radius:8px;'>"
                f"<p><strong>[{evidence_id}]</strong> "
                f"{html.escape(str(item.get('drug', drug)))}</p>"
                f"<p>SmPC section {section} — {title}</p>"
                f"<p>Page(s): {pages} · Chunk ID: {chunk_id}</p>"
                f"<p>{body}</p>"
                f"</article>"
            )

    for drug, items in evidence.items():
        if drug not in DRUG_ORDER and items:
            sections.append(
                f"<p><em>Unlisted drug key in evidence: {html.escape(drug)}</em></p>"
            )

    return "\n".join(sections)


def is_in_scope(query: str) -> bool:
    q = query.lower()

    drug_terms = {
        "jardiance",
        "empagliflozin",
        "forxiga",
        "dapagliflozin",
    }

    smpc_terms = {
        "indication",
        "indications",
        "dose",
        "dosing",
        "dosage",
        "posology",
        "administration",
        "contraindication",
        "contraindications",
        "warning",
        "warnings",
        "precaution",
        "precautions",
        "interaction",
        "interactions",
        "adverse",
        "reaction",
        "reactions",
        "side effect",
        "side effects",
        "ketoacidosis",
        "hypoglycaemia",
        "hypoglycemia",
        "renal",
        "kidney",
        "hepatic",
        "liver",
        "pregnancy",
        "breastfeeding",
        "elderly",
        "paediatric",
        "pediatric",
        "pharmacodynamic",
        "pharmacodynamics",
        "pharmacokinetic",
        "pharmacokinetics",
        "mechanism",
        "efficacy",
        "safety",
        "smpc",
        "section 4",
        "section 5",
    }

    return any(term in q for term in drug_terms | smpc_terms)


def respond(message: str, history: list[dict]):
    query = (message or "").strip()
    history = list(history or [])

    if not query:
        yield history, format_evidence_html(None)
        return

    history.append({"role": "user", "content": query})

    if not is_in_scope(query):
        history.append(
            {
                "role": "assistant",
                "content": (
                    "This system is limited to comparing Jardiance "
                    "(empagliflozin) and Forxiga (dapagliflozin) using EMA "
                    "SmPC evidence. Please ask a question within that scope."
                ),
            }
        )
        yield history, format_evidence_html(None)
        return

    history.append(
        {"role": "assistant", "content": "Retrieving EMA SmPC evidence..."}
    )
    yield history, format_evidence_html(None)

    result = generate_answer(query)
    answer = result["answer"]
    evidence = result["evidence"]

    history[-1] = {"role": "assistant", "content": answer}
    yield history, format_evidence_html(evidence, answer)


def clear_conversation():
    return [], format_evidence_html(None)


def build_app() -> gr.Blocks:
    with gr.Blocks(title="EMA Drug Comparison RAG") as demo:
        gr.Markdown(
            "# EMA Drug Comparison RAG\n\n"
            "Compare **Jardiance** and **Forxiga** using retrieved EMA SmPC evidence."
        )
        gr.Markdown(DISCLAIMER)

        chatbot = gr.Chatbot(
            label = "Conversation",
            height = 420,
        )
        evidence = gr.HTML(
            value = format_evidence_html(None),
            label = "EMA SmPC evidence (latest question)",
        )

        with gr.Row():
            question = gr.Textbox(
                label = "Question",
                placeholder = "Compare the ketoacidosis warnings for Jardiance and Forxiga.",
                scale = 4,
            )
            submit = gr.Button("Ask", variant="primary")

        clear = gr.Button("Clear / new conversation")

        submit.click(
            fn = respond,
            inputs = [question, chatbot],
            outputs = [chatbot, evidence],
        ).then(lambda: "", outputs=question)

        question.submit(
            fn = respond,
            inputs = [question, chatbot],
            outputs = [chatbot, evidence],
        ).then(lambda: "", outputs=question)

        clear.click(
            fn = clear_conversation,
            outputs = [chatbot, evidence],
        )

    return demo


if __name__ == "__main__":
    # Run from the repository root so data/embeddings paths resolve.
    root = Path(__file__).resolve().parent
    if Path.cwd() != root:
        import os

        os.chdir(root)

    build_app().launch(
        server_name="0.0.0.0",
        server_port=7860,
    )
