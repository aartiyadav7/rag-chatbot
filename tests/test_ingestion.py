from src.ingestion.pdf_loader import load_pdf
from src.ingestion.web_loader import load_url
from src.ingestion.chunker import chunk_documents

def test_web_loader():
    pages = load_url("https://en.wikipedia.org/wiki/Retrieval-augmented_generation")
    assert len(pages) > 0
    assert "text" in pages[0]
    print(f"✅ Web loader OK — {len(pages[0]['text'])} chars")

def test_chunker():
    pages = load_url("https://en.wikipedia.org/wiki/Retrieval-augmented_generation")
    chunks = chunk_documents(pages)
    assert len(chunks) > 0
    print(f"✅ Chunker OK — {len(chunks)} chunks created")

if __name__ == "__main__":
    test_web_loader()
    test_chunker()