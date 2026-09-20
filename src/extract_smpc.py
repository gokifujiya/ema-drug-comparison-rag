from pathlib import Path
import json

import pymupdf


RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

DOCUMENTS = {
    "forxiga": {
        "filename": "forxiga_product_information.pdf",
        "drug": "Forxiga",
        "active_substance": "dapagliflozin",
    },
    "jardiance": {
        "filename": "jardiance_product_information.pdf",
        "drug": "Jardiance",
        "active_substance": "empagliflozin",
    },
}


def extract_smpc(pdf_path: Path) -> list[dict]:
    doc = pymupdf.open(pdf_path)

    pages = []
    inside_annex_i = False

    for page_index, page in enumerate(doc):
        text = page.get_text("text")

        if "ANNEX I" in text:
            inside_annex_i = True

        if "ANNEX II" in text:
            break

        if inside_annex_i:
            pages.append(
                {
                    "page": page_index + 1,
                    "text": text.strip(),
                }
            )

    return pages


def main() -> None:
    PROCESSED_DIR.mkdir(parents = True, exist_ok = True)

    for key, metadata in DOCUMENTS.items():
        pdf_path = RAW_DIR / metadata["filename"]

        pages = extract_smpc(pdf_path)

        output = {
            "drug": metadata["drug"],
            "active_substance": metadata["active_substance"],
            "source": "European Medicines Agency (EMA)",
            "document_type": "Summary of Product Characteristics (SmPC)",
            "source_file": metadata["filename"],
            "pages": pages,
        }

        output_path = PROCESSED_DIR / f"{key}_smpc.json"

        with output_path.open("w", encoding = "utf-8") as f:
            json.dump(output, f, ensure_ascii = False, indent = 2)

        print(
            f"{metadata['drug']}: "
            f"extracted {len(pages)} SmPC pages -> {output_path}"
        )


if __name__ == "__main__":
    main()
