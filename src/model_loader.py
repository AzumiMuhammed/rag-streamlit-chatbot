import os
import streamlit as st
from dotenv import load_dotenv
from llama_index.llms.google_genai import GoogleGenAI
from src.config import LLM_MODEL

load_dotenv()

def initialise_llm() -> GoogleGenAI:
    api_key = os.getenv("GOOGLE_API_KEY") or st.secrets.get("GOOGLE_API_KEY", None)
    if not api_key:
        raise ValueError(
            "GOOGLE_API_KEY not found. Add it in Streamlit Cloud → Settings → Secrets."
        )
    return GoogleGenAI(api_key=api_key, model=LLM_MODEL)


from llama_index.embeddings.huggingface import HuggingFaceEmbedding

from src.config import (
    EMBEDDING_MODEL_NAME,
    EMBEDDING_CACHE_PATH,
)

def get_embedding_model() -> HuggingFaceEmbedding:
    """Initialises and returns the HuggingFace embedding model"""
    # Create the cache directory if it doesn't exist
    EMBEDDING_CACHE_PATH.mkdir(parents=True, exist_ok=True)

    return HuggingFaceEmbedding(
        model_name=EMBEDDING_MODEL_NAME,
        cache_folder=EMBEDDING_CACHE_PATH.as_posix()
    )
