import fitz
from pathlib import Path
from src.utils.logger import get_logger

logger = get_logger(__name__)

def load_pdf(file_path: str) -> list[dict]:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"PDF not found: {file_path}")

    logger.info(f"Loading PDF: {path.name}")
    doc = fitz.open(str(path))
    pages = []

    for page_num, page in enumerate(doc):
        text = page.get_text().strip()
        if text:
            pages.append({
                "page": page_num + 1,
                "text": text,
                "source": path.name
            })

    doc.close()
    logger.info(f"Extracted {len(pages)} pages from {path.name}")
    return pages