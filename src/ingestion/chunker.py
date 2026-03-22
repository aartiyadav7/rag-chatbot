from langchain_text_splitters import RecursiveCharacterTextSplitter
from src.utils.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

def chunk_documents(pages: list[dict]) -> list[dict]:
    """
    Split pages into smaller overlapping chunks for embedding.
    Each chunk: { "text": str, "source": str, "page": int, "chunk_id": int }
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ".", " ", ""]
    )

    all_chunks = []
    chunk_id = 0

    for page in pages:
        splits = splitter.split_text(page["text"])
        for split in splits:
            all_chunks.append({
                "chunk_id": chunk_id,
                "text": split,
                "source": page["source"],
                "page": page["page"]
            })
            chunk_id += 1

    logger.info(f"Created {len(all_chunks)} chunks from {len(pages)} pages")
    return all_chunks