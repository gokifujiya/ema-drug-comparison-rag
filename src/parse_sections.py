from pathlib import Path
import json
import re


PROCESSED_DIR = Path("data/processed")


EXPECTED_SECTIONS = [
    ("1", "NAME OF THE MEDICINAL PRODUCT"),
    ("2", "QUALITATIVE AND QUANTITATIVE COMPOSITION"),
    ("3", "PHARMACEUTICAL FORM"),
    ("4", "CLINICAL PARTICULARS"),
    ("4.1", "Therapeutic indications"),
    ("4.2", "Posology and method of administration"),
    ("4.3", "Contraindications"),
    ("4.4", "Special warnings and precautions for use"),
    (
        "4.5",
        "Interaction with other medicinal products and other forms of interaction",
    ),
    ("4.6", "Fertility, pregnancy and lactation"),
    ("4.7", "Effects on ability to drive and use machines"),
    ("4.8", "Undesirable effects"),
    ("4.9", "Overdose"),
    ("5", "PHARMACOLOGICAL PROPERTIES"),
    ("5.1", "Pharmacodynamic properties"),
    ("5.2", "Pharmacokinetic properties"),
    ("5.3", "Preclinical safety data"),
    ("6", "PHARMACEUTICAL PARTICULARS"),
    ("6.1", "List of excipients"),
    ("6.2", "Incompatibilities"),
    ("6.3", "Shelf life"),
    ("6.4", "Special precautions for storage"),
    ("6.5", "Nature and contents of container"),
    ("6.6", "Special precautions for disposal"),
]


def load_smpc(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def combine_pages(data: dict) -> str:
    parts = []

    for page in data["pages"]:
        parts.append(
            f"\n[[PAGE {page['page']}]]\n{page['text']}"
        )

    return "\n".join(parts)


def find_section(text: str, number: str, title: str):
    """
    Find a genuine SmPC section using both its section number
    and expected official heading.
    """
    pattern = re.compile(
        rf"(?m)^{re.escape(number)}\.?\s*\n"
        rf"{re.escape(title)}\s*$",
        re.IGNORECASE,
    )

    return pattern.search(text)


def pages_in_text(text: str) -> list[int]:
    markers = re.findall(r"\[\[PAGE (\d+)\]\]", text)
    return sorted({int(page) for page in markers})


def page_at_position(text: str, position: int) -> int | None:
    markers = list(
        re.finditer(r"\[\[PAGE (\d+)\]\]", text[:position])
    )

    if not markers:
        return None

    return int(markers[-1].group(1))


def parse_sections(text: str) -> list[dict]:
    found = []

    # Locate all genuine SmPC headings.
    for number, title in EXPECTED_SECTIONS:
        match = find_section(text, number, title)

        if match:
            found.append(
                {
                    "section": number,
                    "title": title,
                    "start": match.start(),
                    "content_start": match.end(),
                }
            )
        else:
            print(f"WARNING: section {number} not found: {title}")

    # Sort according to actual position in the document.
    found.sort(key = lambda item: item["start"])

    sections = []

    for i, item in enumerate(found):
        if i + 1 < len(found):
            end = found[i + 1]["start"]
        else:
            end = len(text)

        content = text[item["content_start"]:end].strip()

        section_text = text[item["start"]:end]

        pages = pages_in_text(section_text)

        start_page = page_at_position(text, item["start"])

        if start_page is not None:
            pages = sorted(set([start_page] + pages))

        sections.append(
            {
                "section": item["section"],
                "title": item["title"],
                "pages": pages,
                "text": content,
            }
        )

    return sections


def main() -> None:
    for path in sorted(PROCESSED_DIR.glob("*_smpc.json")):
        data = load_smpc(path)

        text = combine_pages(data)
        sections = parse_sections(text)

        output = {
            "drug": data["drug"],
            "active_substance": data["active_substance"],
            "source": data["source"],
            "document_type": data["document_type"],
            "source_file": data["source_file"],
            "sections": sections,
        }

        output_path = PROCESSED_DIR / path.name.replace(
            "_smpc.json",
            "_sections.json",
        )

        with output_path.open("w", encoding = "utf-8") as f:
            json.dump(output, f, ensure_ascii = False, indent = 2)

        print(f"\n{data['drug']}: {len(sections)} sections")

        for section in sections:
            print(
                f"{section['section']:>4}  "
                f"{section['title']:<70} "
                f"pages={section['pages']}"
            )

        print(f"\nSaved -> {output_path}")


if __name__ == "__main__":
    main()
