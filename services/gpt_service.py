"""
Backward-compatibility adapter for services expecting gpt_service.
Routes all LLM generation requests to generate_with_ollama.
"""
from typing import Optional
from services.ollama_service import generate_with_ollama

def generate_with_gpt(prompt: str, question: Optional[str] = None) -> str:
    """
    Adapter function mapping generate_with_gpt to generate_with_ollama.
    """
    return generate_with_ollama(prompt, question)
