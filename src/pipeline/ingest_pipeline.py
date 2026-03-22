from src.ingestion.pdf_loader import load_pdf
from src.ingestion.web_loader import load_url
from src.ingestion.chunker import chunk_documents
from src.embeddings.embedder import embed_chunks
from src.retrieval.vector_store import build_vector_store
from src.utils.logger import get_logger

logger = get_logger(__name__)

def ingest_pdf(file_path: str, reset: bool = True):
    """Full ingestion pipeline for a PDF file."""
    logger.info(f"Ingesting PDF: {file_path}")
    pages = load_pdf(file_path)
    chunks = chunk_documents(pages)
    texts, metadatas = embed_chunks(chunks)
    vector_store = build_vector_store(texts, metadatas, reset=reset)
    logger.info(f"PDF ingested: {len(chunks)} chunks stored")
    return vector_store

def ingest_url(url: str, reset: bool = True):
    """Full ingestion pipeline for a URL."""
    logger.info(f"Ingesting URL: {url}")
    pages = load_url(url)
    chunks = chunk_documents(pages)
    texts, metadatas = embed_chunks(chunks)
    vector_store = build_vector_store(texts, metadatas, reset=reset)
    logger.info(f"URL ingested: {len(chunks)} chunks stored")
    return vector_store