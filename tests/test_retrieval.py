from src.ingestion.web_loader import load_url
from src.ingestion.chunker import chunk_documents
from src.embeddings.embedder import embed_chunks
from src.retrieval.vector_store import build_vector_store, retrieve_chunks

def test_vector_store():
    # Load & chunk
    pages = load_url("https://en.wikipedia.org/wiki/Retrieval-augmented_generation")
    chunks = chunk_documents(pages)

    # Embed & store (reset=True cleans old db before building)
    texts, metadatas = embed_chunks(chunks)
    vector_store = build_vector_store(texts, metadatas, reset=True)

    # Retrieve
    results = retrieve_chunks("What is RAG?", vector_store)
    assert len(results) > 0
    assert "text" in results[0]
    print(f"\n✅ Vector store OK — retrieved {len(results)} chunks")
    print(f"📄 Top chunk preview: {results[0]['text'][:200]}")

if __name__ == "__main__":
    test_vector_store()