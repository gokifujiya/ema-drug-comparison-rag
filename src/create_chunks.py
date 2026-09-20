from pathlib import Path
import json
import re


PROCESSED_DIR = Path("data/processed")

MAX_CHARS = 3500
OVERLAP_CHARS = 400


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def clean_text(text: str) -> str:
    # Remove our internal page markers from the text.
    text = re.sub(r"\[\[PAGE \d+\]\]", "", text)

    # Normalize excessive blank lines.
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def pages_in_text(text: str) -> list[int]:
    """Return page numbers found in [[PAGE N]] markers."""
    return sorted(
        {
            int(page)
            for page in re.findall(r"\[\[PAGE (\d+)\]\]", text)
        }
    )


def page_before_position(text: str, position: int) -> int | None:
    """
    Return the most recent page marker before a character position.
    This handles chunks that begin after a page marker.
    """
    markers = list(
        re.finditer(
            r"\[\[PAGE (\d+)\]\]",
            text[:position],
        )
    )

    if not markers:
        return None

    return int(markers[-1].group(1))


def split_long_text(
    text: str,
    initial_page: int | None = None,
    max_chars: int = MAX_CHARS,
    overlap: int = OVERLAP_CHARS,
) -> list[dict]:
    """
    Split long section text while preserving exact page provenance.
    """

    chunks = []
    start = 0

    while start < len(text):
        target_end = min(start + max_chars, len(text))

        if target_end == len(text):
            end = target_end
        else:
            window = text[start:target_end]

            # Prefer paragraph boundary.
            boundary = window.rfind("\n\n")

            # Otherwise try sentence boundary.
            if boundary < max_chars // 2:
                boundary = window.rfind(". ")

            if boundary < max_chars // 2:
                end = target_end
            else:
                end = start + boundary + 1

        raw_chunk = text[start:end]

        pages = pages_in_text(raw_chunk)

        # A chunk may begin after its page marker.
        start_page = page_before_position(text, start)

        if start_page is None:
            start_page = initial_page

        if start_page is not None:
            pages = sorted(set([start_page] + pages))

        chunk_text = clean_text(raw_chunk)

        if chunk_text:
            chunks.append(
                {
                    "text": chunk_text,
                    "pages": pages,
                }
            )

        if end >= len(text):
            break

        # Create overlap.
        new_start = max(end - overlap, start + 1)

        # Avoid beginning in the middle of a word.
        while (
            new_start < end
            and not text[new_start].isspace()
        ):
            new_start += 1

        while (
            new_start < end
            and text[new_start].isspace()
        ):
            new_start += 1

        start = new_start

    return chunks


def create_chunks(data: dict) -> list[dict]:
    chunks = []

    for section in data["sections"]:
        text = section["text"]

        if not clean_text(text):
            continue

        initial_page = (
            section["pages"][0]
            if section["pages"]
            else None
        )

        pieces = split_long_text(
            text,
            initial_page = initial_page,
        )

        for index, piece in enumerate(pieces, start=1):
            chunk_id = (
                f"{data['drug'].lower()}_"
                f"smpc_{section['section'].replace('.', '_')}_"
                f"{index:03d}"
            )

            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "drug": data["drug"],
                    "active_substance": data["active_substance"],
                    "source": data["source"],
                    "document_type": data["document_type"],
                    "source_file": data["source_file"],
                    "section": section["section"],
                    "section_title": section["title"],
                    "pages": piece["pages"],
                    "chunk_index": index,
                    "text": piece["text"],
                }
            )

    return chunks


def main() -> None:
    for path in sorted(PROCESSED_DIR.glob("*_sections.json")):
        data = load_json(path)

        chunks = create_chunks(data)

        output = {
            "drug": data["drug"],
            "active_substance": data["active_substance"],
            "document_type": data["document_type"],
            "chunks": chunks,
        }

        output_path = PROCESSED_DIR / path.name.replace(
            "_sections.json",
            "_chunks.json",
        )

        with output_path.open("w", encoding = "utf-8") as f:
            json.dump(output, f, ensure_ascii = False, indent = 2)

        print(f"\n{data['drug']}")
        print(f"Sections: {len(data['sections'])}")
        print(f"Chunks:   {len(chunks)}")

        print("\nChunks by section:")

        section_counts = {}

        for chunk in chunks:
            section = chunk["section"]
            section_counts[section] = section_counts.get(section, 0) + 1

        for section, count in section_counts.items():
            print(f"  {section:>4}: {count} chunk(s)")

        print(f"\nSaved -> {output_path}")


if __name__ == "__main__":
    main()
