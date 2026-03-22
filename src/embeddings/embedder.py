from langchain_huggingface import HuggingFaceEmbeddings
from src.utils.logger import get_logger

logger = get_logger(__name__)

EMBEDDING_MODEL = "all-MiniLM-L6-v2"  # free, fast, no API key needed

def get_embedder() -> HuggingFaceEmbeddings:
    """Returns a free local HuggingFace embeddings instance."""
    logger.info(f"Loading local embedder: {EMBEDDING_MODEL}")
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )

def embed_chunks(chunks: list[dict]) -> tuple[list[str], list[dict]]:
    """Extract texts and metadata from chunks for vector store ingestion."""
    texts = [chunk["text"] for chunk in chunks]
    metadatas = [
        {
            "source": chunk["source"],
            "page": chunk["page"],
            "chunk_id": chunk["chunk_id"]
        }
        for chunk in chunks
    ]
    logger.info(f"Prepared {len(texts)} texts for embedding")
    return texts, metadatas