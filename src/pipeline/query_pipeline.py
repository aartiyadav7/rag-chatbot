from src.retrieval.vector_store import load_vector_store, retrieve_chunks
from src.generation.llm import generate_answer
from src.utils.logger import get_logger

logger = get_logger(__name__)

def run_query(question: str) -> dict:
    """
    Full RAG query pipeline:
    1. Load vector store
    2. Retrieve relevant chunks
    3. Generate answer with LLM
    """
    logger.info(f"Running query: {question}")
    vector_store = load_vector_store()
    chunks = retrieve_chunks(question, vector_store)
    result = generate_answer(question, chunks)
    return {
        "answer": result["answer"],
        "sources": result["sources"],
        "chunks": chunks
    }