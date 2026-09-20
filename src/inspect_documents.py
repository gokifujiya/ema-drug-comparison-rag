from pathlib import Path

import pymupdf


RAW_DIR = Path("data/raw")


def inspect_pdf(pdf_path: Path) -> None:
    doc = pymupdf.open(pdf_path)

    print("=" * 80)
    print(f"File: {pdf_path.name}")
    print(f"Pages: {len(doc)}")
    print("=" * 80)

    for page_number, page in enumerate(doc, start=1):
        text = page.get_text("text")

        if "ANNEX I" in text or "ANNEX II" in text or "ANNEX III" in text:
            print(f"\nPage {page_number}")
            
            for line in text.splitlines():
                if "ANNEX" in line:
                    print(f"  {line}")


def main() -> None:
    for pdf_path in sorted(RAW_DIR.glob("*.pdf")):
        inspect_pdf(pdf_path)


if __name__ == "__main__":
    main()
