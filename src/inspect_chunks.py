from pathlib import Path
import json


PROCESSED_DIR = Path("data/processed")


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def main() -> None:
    for path in sorted(PROCESSED_DIR.glob("*_chunks.json")):
        data = load_json(path)

        print("\n" + "=" * 80)
        print(data["drug"])
        print("=" * 80)

        # Inspect particularly important sections.
        for section_number in ["4.2", "4.4", "4.8", "5.1"]:
            print(f"\n--- SECTION {section_number} ---")

            chunks = [
                chunk
                for chunk in data["chunks"]
                if chunk["section"] == section_number
            ]

            for chunk in chunks:
                text = chunk["text"]

                print(
                    f"\n{chunk['chunk_id']} | "
                    f"{len(text)} chars | "
                    f"pages={chunk['pages']}"
                )

                # Only show beginning/end to keep terminal manageable.
                print("START:", repr(text[:180]))
                print("END:  ", repr(text[-180:]))


if __name__ == "__main__":
    main()
