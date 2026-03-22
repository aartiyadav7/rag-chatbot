from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage
from src.generation.prompt_templates import build_prompt
from src.utils.config import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

def get_llm() -> ChatGroq:
    """Returns a Groq LLM instance."""
    model = "llama-3.1-8b-instant"  
    groq_key = settings.groq_api_key
    logger.info(f"Loading LLM: {model}")
    return ChatGroq(
        model=model,
        groq_api_key=groq_key,
        temperature=0.2,
        max_tokens=1024
    )

def generate_answer(question: str, chunks: list[dict]) -> dict:
    """
    Generate an answer from retrieved chunks using Groq LLM.
    Returns: { "answer": str, "sources": list }
    """
    if not chunks:
        return {
            "answer": "I could not find relevant information to answer your question.",
            "sources": []
        }

    system_prompt, user_prompt = build_prompt(question, chunks)
    llm = get_llm()

    logger.info(f"Generating answer for: {question[:50]}...")
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ]

    response = llm.invoke(messages)
    sources = list(set([
        f"{c['source']} (page {c['page']})" for c in chunks
    ]))

    logger.info("Answer generated successfully")
    return {
        "answer": response.content,
        "sources": sources
    }