RAG_SYSTEM_PROMPT = """You are a helpful AI assistant that answers questions 
based strictly on the provided context. 

Rules:
- Only use information from the context below to answer
- If the answer is not in the context, say "I don't have enough information to answer this"
- Always mention which source your answer came from
- Be concise and clear
"""

RAG_USER_PROMPT = """Context:
{context}

Question: {question}

Answer based only on the context above:"""

def build_prompt(question: str, chunks: list[dict]) -> tuple[str, str]:
    context_parts = []
    for i, chunk in enumerate(chunks):
        context_parts.append(
            f"[Source {i+1}: {chunk['source']} | Page {chunk['page']}]\n{chunk['text']}"
        )
    context = "\n\n---\n\n".join(context_parts)
    user_prompt = RAG_USER_PROMPT.format(context=context, question=question)
    return RAG_SYSTEM_PROMPT, user_prompt