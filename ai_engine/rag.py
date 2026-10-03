from typing import Optional
from services.pinecone_service import similarity_search
from services.ollama_service import generate_with_ollama

RAG_PROMPT = """You are a maternal health assistant helping women during pregnancy
and the first 2 years of the baby's life.

Answer the question ONLY using the information from the context.

If the answer is not present in the context, say:
"I couldn't find this information in the document."

CONTEXT:
{context}

QUESTION:
{question}

Give a clear and well-formatted helpful answer based strictly on the context.
"""

RAG_PROMPT_WITH_USER_CONTEXT = """You are a maternal health assistant helping women during pregnancy
and the first 2 years of the baby's life.

USER PROFILE / CONTEXT:
{user_context}

Answer the question ONLY using the information from the guidance context below. Use the user profile/context for personalization, stage awareness, and tone, but answer strictly and factually based on the guidance context.

If the answer is not present in the context, say:
"I couldn't find this information in the document."

CONTEXT FROM GUIDANCE:
{context}

QUESTION:
{question}

Give a clear and well-formatted helpful answer based strictly on the context.
"""


def run_rag(vectorstore, question: str, user_context: Optional[str] = None) -> dict:
    results = similarity_search(vectorstore, question, k=12)

    context = "\n---\n".join([content for content, _ in results])

    if user_context and user_context.strip():
        prompt = RAG_PROMPT_WITH_USER_CONTEXT.format(
            user_context=user_context.strip(),
            context=context,
            question=question
        )
    else:
        prompt = RAG_PROMPT.format(
            context=context,
            question=question
        )

    answer = generate_with_ollama(prompt, question, user_context=user_context)

    return {
        "answer": answer,
        "context": context
    }
