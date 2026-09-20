from pathlib import Path

import requests


DOCUMENTS = {
    "jardiance_product_information.pdf": (
        "https://www.ema.europa.eu/en/documents/product-information/"
        "jardiance-epar-product-information_en.pdf"
    ),
    "forxiga_product_information.pdf": (
        "https://www.ema.europa.eu/en/documents/product-information/"
        "forxiga-epar-product-information_en.pdf"
    ),
}

RAW_DATA_DIR = Path("data/raw")


def download_document(url: str, output_path: Path) -> None:
    response = requests.get(url, timeout=60)
    response.raise_for_status()

    output_path.write_bytes(response.content)
    print(f"Downloaded: {output_path}")


def main() -> None:
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    for filename, url in DOCUMENTS.items():
        download_document(url, RAW_DATA_DIR / filename)


if __name__ == "__main__":
    main()
