import shutil
import os
import time
import gc
import chromadb
from langchain_chroma import Chroma
from src.embeddings.embedder import get_embedder
from src.utils.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

_chroma_client = None
_vector_store = None

def _close_chroma_client():
    """Force close all ChromaDB connections and release file handles."""
    global _chroma_client, _vector_store

    # Close vector store first
    if _vector_store is not None:
        try:
            del _vector_store
        except Exception:
            pass
        _vector_store = None

    # Close client
    if _chroma_client is not None:
        try:
            _chroma_client.clear_system_cache()
        except Exception:
            pass
        try:
            del _chroma_client
        except Exception:
            pass
        _chroma_client = None

    # Force Python garbage collection
    gc.collect()
    time.sleep(1.5)

def _get_chroma_client():
    global _chroma_client
    if _chroma_client is None:
        _chroma_client = chromadb.PersistentClient(
            path=settings.chroma_db_path
        )
    return _chroma_client

def _safe_delete_chroma():
    """Delete ChromaDB folder safely on Windows."""
    if not os.path.exists(settings.chroma_db_path):
        return

    _close_chroma_client()

    for attempt in range(8):
        try:
            # Try renaming first — Windows trick to break file locks
            temp_path = settings.chroma_db_path + "_old"
            if os.path.exists(temp_path):
                shutil.rmtree(temp_path, ignore_errors=True)

            os.rename(settings.chroma_db_path, temp_path)
            shutil.rmtree(temp_path, ignore_errors=True)
            logger.info("Removed existing vector store folder")
            return
        except Exception as e:
            if attempt < 7:
                logger.info(f"Retrying delete ({attempt+1}/8)...")
                gc.collect()
                time.sleep(1.5)
            else:
                # Last resort — delete file by file
                try:
                    _force_delete_folder(settings.chroma_db_path)
                    return
                except Exception as final_e:
                    raise RuntimeError(
                        f"Could not delete ChromaDB. "
                        f"Stop the app, manually delete the folder "
                        f"'data/processed/chroma_db', then restart.\n"
                        f"Error: {final_e}"
                    )

def _force_delete_folder(path: str):
    """Delete folder file by file, ignoring locked files."""
    for root, dirs, files in os.walk(path, topdown=False):
        for file in files:
            try:
                fp = os.path.join(root, file)
                os.chmod(fp, 0o777)
                os.remove(fp)
            except Exception:
                pass
        for d in dirs:
            try:
                os.rmdir(os.path.join(root, d))
            except Exception:
                pass
    try:
        os.rmdir(path)
    except Exception:
        pass
    logger.info("Force deleted vector store folder")

def build_vector_store(
    texts: list[str],
    metadatas: list[dict],
    reset: bool = False
) -> Chroma:
    """Create and persist a ChromaDB vector store from texts."""
    global _vector_store

    _safe_delete_chroma()
    os.makedirs(settings.chroma_db_path, exist_ok=True)
    logger.info(f"Building vector store with {len(texts)} chunks...")

    embedder = get_embedder()
    client = _get_chroma_client()

    _vector_store = Chroma.from_texts(
        texts=texts,
        embedding=embedder,
        metadatas=metadatas,
        client=client,
        collection_name="rag_collection"
    )
    logger.info(f"Vector store built with {len(texts)} chunks")
    return _vector_store

def load_vector_store() -> Chroma:
    """Load an existing ChromaDB vector store from disk."""
    logger.info(f"Loading vector store from {settings.chroma_db_path}")
    embedder = get_embedder()
    client = _get_chroma_client()
    return Chroma(
        client=client,
        collection_name="rag_collection",
        embedding_function=embedder
    )

def retrieve_chunks(query: str, vector_store: Chroma) -> list[dict]:
    """Retrieve top-k relevant chunks for a query."""
    logger.info(f"Retrieving top {settings.top_k} chunks for query...")
    results = vector_store.similarity_search(query, k=settings.top_k)
    chunks = [
        {
            "text": doc.page_content,
            "source": doc.metadata.get("source", "unknown"),
            "page": doc.metadata.get("page", 0)
        }
        for doc in results
    ]
    logger.info(f"Retrieved {len(chunks)} chunks")
    return chunks