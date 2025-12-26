import streamlit as st

@st.cache_resource(show_spinner=False)
def get_embedder(model_name: str):
    """
    Load the embedding model only once per app instance.
    This prevents Streamlit Cloud from re-loading it on every rerun.
    """
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_name)
