from pathlib import Path

from docx import Document
from reportlab.pdfgen import canvas

from app.rag.document_loader import find_documents, load_document, split_pages
from app.rag.ingestion import ingest_documents
from app.rag.retriever import Retriever
from app.rag.vector_store import VectorStore


def make_documents(directory: Path) -> None:
    (directory / "faq.txt").write_text("Office hours are nine to five on weekdays.")
    (directory / "guide.md").write_text("# Guide\nPull requests need one approval before merging.")
    doc = Document()
    doc.add_paragraph("Employees receive twenty five days of annual leave.")
    doc.save(directory / "handbook.docx")
    pdf = canvas.Canvas(str(directory / "policy.pdf"))
    pdf.drawString(72, 750, "Laptops must use full disk encryption.")
    pdf.showPage()
    pdf.drawString(72, 750, "Flights shorter than six hours are economy class.")
    pdf.save()
    (directory / "ignored.exe").write_text("not a document")


def test_loader_reads_every_supported_format(tmp_path: Path) -> None:
    make_documents(tmp_path)
    paths = find_documents(tmp_path)
    assert [p.name for p in paths] == ["faq.txt", "guide.md", "handbook.docx", "policy.pdf"]

    pages = {p.name: load_document(p) for p in paths}
    assert "nine to five" in pages["faq.txt"][0].text
    assert "approval" in pages["guide.md"][0].text
    assert "annual leave" in pages["handbook.docx"][0].text
    assert [page.page for page in pages["policy.pdf"]] == [1, 2]


def test_chunking_respects_size_and_keeps_metadata(tmp_path: Path) -> None:
    text = " ".join(f"word{i}" for i in range(500))
    (tmp_path / "long.txt").write_text(text)
    chunks = split_pages(load_document(tmp_path / "long.txt"), chunk_size=200, chunk_overlap=20)

    assert len(chunks) > 1
    assert all(len(c.text) <= 200 for c in chunks)
    assert [c.chunk for c in chunks] == list(range(len(chunks)))
    assert chunks[0].metadata == {"source": "long.txt", "type": "txt", "chunk": 0}


def test_retrieval_returns_relevant_chunk_with_page(tmp_path: Path, embedder) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    make_documents(docs)
    store = VectorStore(tmp_path / "chroma", embedder)

    result = ingest_documents(store, docs, chunk_size=500, chunk_overlap=50)
    assert result.documents == 4
    assert result.chunks == store.count() > 0

    hits = Retriever(store, top_k=2, max_distance=1.0).retrieve("flights economy class")
    assert hits[0].source == "policy.pdf"
    assert hits[0].page == 2


def test_retriever_filters_irrelevant_chunks(tmp_path: Path, embedder) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "faq.txt").write_text("Office hours are nine to five on weekdays.")
    store = VectorStore(tmp_path / "chroma", embedder)
    ingest_documents(store, docs, chunk_size=500, chunk_overlap=50)

    assert Retriever(store, top_k=3, max_distance=0.5).retrieve("quantum chromodynamics") == []


def test_rebuild_replaces_previous_index(tmp_path: Path, embedder) -> None:
    docs = tmp_path / "docs"
    docs.mkdir()
    (docs / "a.txt").write_text("alpha")
    store = VectorStore(tmp_path / "chroma", embedder)
    ingest_documents(store, docs, 500, 50)
    (docs / "a.txt").unlink()
    (docs / "b.txt").write_text("beta")
    ingest_documents(store, docs, 500, 50)

    assert store.count() == 1
