from hr_rag.ingestion.chunking import chunk_sections, split_text, title_from_source
from hr_rag.ingestion.loaders import Section


def test_short_text_is_single_chunk():
    assert split_text("Annual leave is 20 days.", 100, 20) == ["Annual leave is 20 days."]


def test_chunks_respect_size_and_overlap():
    paragraphs = [f"Paragraph {i}. " + "word " * 40 for i in range(20)]
    text = "\n\n".join(paragraphs)
    chunks = split_text(text, chunk_size=400, chunk_overlap=80)
    assert len(chunks) > 1
    assert all(len(c) <= 400 for c in chunks)
    # Consecutive chunks share overlapping text.
    assert chunks[1].split()[0] in chunks[0]


def test_giant_word_is_hard_split():
    chunks = split_text("x" * 1000, chunk_size=300, chunk_overlap=50)
    assert all(len(c) <= 300 for c in chunks)
    assert chunks[0] == "x" * 300
    assert sum(len(c) for c in chunks) >= 1000  # nothing lost; overlap repeats some text


def test_chunk_ids_are_deterministic_and_keep_pages():
    sections = [Section("Leave policy text. " * 50, 1), Section("Benefits text. " * 50, 2)]
    a = chunk_sections(sections, "gs://b/leave_policy.pdf", 300, 50)
    b = chunk_sections(sections, "gs://b/leave_policy.pdf", 300, 50)
    assert [c.id for c in a] == [c.id for c in b]
    assert {c.page for c in a} == {1, 2}
    assert a[0].title == "Leave Policy"
    assert a[0].embedding_text().startswith("Document: Leave Policy (page 1)")


def test_title_from_source():
    assert title_from_source("gs://bucket/hr/remote-work_policy.docx") == "Remote Work Policy"
