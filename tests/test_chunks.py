import json
from pathlib import Path


CHUNKS_FILE = Path("data/embeddings/chunks.json")


def load_chunks():
    with open(CHUNKS_FILE, encoding="utf-8") as f:
        return json.load(f)


def test_total_chunk_count():
    chunks = load_chunks()
    assert len(chunks) == 99


def test_both_drugs_present():
    chunks = load_chunks()

    drugs = {chunk["drug"] for chunk in chunks}

    assert "Jardiance" in drugs
    assert "Forxiga" in drugs


def test_chunk_ids_are_unique():
    chunks = load_chunks()

    chunk_ids = [chunk["chunk_id"] for chunk in chunks]

    assert len(chunk_ids) == len(set(chunk_ids))


def test_all_chunks_have_pages():
    chunks = load_chunks()

    for chunk in chunks:
        assert chunk["pages"], (
            f"{chunk['chunk_id']} has no page provenance"
        )


def test_required_metadata_present():
    chunks = load_chunks()

    required_fields = {
        "chunk_id",
        "drug",
        "active_substance",
        "source",
        "document_type",
        "source_file",
        "section",
        "section_title",
        "pages",
        "chunk_index",
        "text",
    }

    for chunk in chunks:
        assert required_fields.issubset(chunk.keys()), (
            f"Missing metadata in {chunk.get('chunk_id')}"
        )


def test_expected_chunk_counts_per_drug():
    chunks = load_chunks()

    jardiance = [
        chunk for chunk in chunks
        if chunk["drug"] == "Jardiance"
    ]

    forxiga = [
        chunk for chunk in chunks
        if chunk["drug"] == "Forxiga"
    ]

    assert len(jardiance) == 48
    assert len(forxiga) == 51


def test_section_4_2_exists_for_both_drugs():
    chunks = load_chunks()

    for drug in ("Jardiance", "Forxiga"):
        matches = [
            chunk for chunk in chunks
            if chunk["drug"] == drug
            and chunk["section"] == "4.2"
        ]

        assert matches, f"No section 4.2 chunk for {drug}"
