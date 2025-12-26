import streamlit as st

from src.model_loader import initialise_llm, get_embedding_model
from src.engine import get_chat_engine


# ---------------------------------------------------------
# Basic page config
# ---------------------------------------------------------
st.set_page_config(
    page_title="ImmiExpert",
    page_icon="🤖✨",
    layout="wide",
)

st.markdown("""
**Tech Stack:** Streamlit · LlamaIndex · Google Gemini · RAG · Vector Search  

This chatbot answers questions about German immigration using
retrieval-augmented generation over curated documents.
""")

st.markdown("""
**Sample questions:**
- What are the requirements for a German Blue Card?
- What are the requirements for German settlement permit?
- Can students work part-time in Germany?
""")


# ---------------------------------------------------------
# Source → URL mapping (for your PDFs / docs)
# ---------------------------------------------------------
SOURCE_URL_MAP: dict[str, tuple[str, str]] = {
    "entry": (
        "Entry regulations for Germany",
        "https://www.bamf.de/EN/Startseite/startseite_node.html",
    ),
    "bamf_bundesamt für migration und flüchtlinge _entry regulations": (
        "BAMF – Entry regulations",
        "https://www.bamf.de/EN/Startseite/startseite_node.html",
    ),
    "bmwk_fast-track_procedure_for_skilled_workers_2024_en-bf": (
        "Fast-track procedure for skilled workers",
        "https://www.make-it-in-germany.com/en/looking-for-foreign-professionals/entering/the-fast-track-procedure-for-skilled-workers",
    ),
    "housing_registration": (
        "Housing registration",
        "https://www.make-it-in-germany.com/en/",
    ),
    "study-visa": (
        "Study visas (search)",
        "https://www.make-it-in-germany.com/en/search?tx_solr%5Bq%5D=study+visas",
    ),
    "family": (
        "Visa for Family Reunification (skilled workers)",
        "https://www.make-it-in-germany.com/en/",
    ),
    "facts": (
        "Facts about Germany",
        "https://www.tatsachen-ueber-deutschland.de/en",
    ),
    "immigration": (
        "About German immigration",
        "https://www.make-it-in-germany.com/en/living-in-germany/discover-germany/immigration",
    ),
    "settlement-permit": (
        "Settlement permit",
        "https://www.make-it-in-germany.com/en/visa-residence/living-permanently/settlement-permit",
    ),
    "working": (
        "Working in Germany",
        "https://www.tatsachen-ueber-deutschland.de/en",
    ),
    "german citizenship": (
        "German citizenship",
        "https://www.tatsachen-ueber-deutschland.de/en/migration-and-integration/more-inclusion-thanks-citizenship",
    ),
}


def map_source_to_url(file_name: str) -> tuple[str, str | None]:
    lower_name = (file_name or "").lower()
    for key, (label, url) in SOURCE_URL_MAP.items():
        if key in lower_name:
            return label, url
    short_name = (file_name or "").split("/")[-1]
    return short_name, None


def extract_source_links(response) -> list[dict]:
    source_links: list[dict] = []
    seen: set[str] = set()

    for node in getattr(response, "source_nodes", []) or []:
        meta = getattr(node, "metadata", {}) or {}
        file_name = (
            meta.get("file_name")
            or meta.get("filename")
            or meta.get("source")
            or meta.get("doc_id")
        )

        if not file_name or file_name in seen:
            continue

        seen.add(file_name)
        label, url = map_source_to_url(str(file_name))
        source_links.append({"label": label, "url": url, "raw": str(file_name)})

    return source_links


# ---------------------------------------------------------
# IMPORTANT: Lazy-load heavy resources (prevents Streamlit Cloud health-check EOF)
# ---------------------------------------------------------
@st.cache_resource
def _cached_llm():
    return initialise_llm()


@st.cache_resource
def _cached_embedder():
    return get_embedding_model()


def _build_chat_engine(similarity_top_k: int):
    """
    Build chat engine, trying to pass similarity_top_k if your get_chat_engine supports it.
    Falls back gracefully if your current signature doesn't accept it.
    """
    llm = _cached_llm()
    embed_model = _cached_embedder()

    try:
        # If your src.engine.get_chat_engine supports similarity_top_k, this will work
        return get_chat_engine(llm, embed_model, similarity_top_k=similarity_top_k)
    except TypeError:
        # Backwards-compatible: old signature get_chat_engine(llm, embed_model)
        return get_chat_engine(llm, embed_model)


def ensure_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "similarity_top_k" not in st.session_state:
        st.session_state.similarity_top_k = 4
    # IMPORTANT: Do NOT create chat engine at startup; only when needed.
    if "chat_engine" not in st.session_state:
        st.session_state.chat_engine = None
    if "chat_engine_top_k" not in st.session_state:
        st.session_state.chat_engine_top_k = None


def get_or_create_engine() -> object:
    """
    Create engine only when needed (first question, or after settings change).
    """
    top_k = int(st.session_state.similarity_top_k)

    needs_new = (
        st.session_state.chat_engine is None
        or st.session_state.chat_engine_top_k is None
        or st.session_state.chat_engine_top_k != top_k
    )

    if needs_new:
        st.session_state.chat_engine = _build_chat_engine(similarity_top_k=top_k)
        st.session_state.chat_engine_top_k = top_k

    return st.session_state.chat_engine


# ---------------------------------------------------------
# Main app
# ---------------------------------------------------------
def main() -> None:
    ensure_session_state()

    # Header
    st.markdown(
        "<h1 style='color:#e63946; margin-bottom:0.3rem;'>🤖✨ ImmiExpert</h1>",
        unsafe_allow_html=True,
    )
    st.markdown(
        "<p style='color:#e63946; font-size:0.95rem; margin-top:0;'>"
        "Your 🇩🇪 immigration information centre"
        "</p>",
        unsafe_allow_html=True,
    )

    # Sidebar
    with st.sidebar:
        st.markdown("### ⚙️ Settings")

        st.session_state.similarity_top_k = st.slider(
            "Number of chunks to retrieve",
            min_value=2,
            max_value=10,
            value=int(st.session_state.similarity_top_k),
            step=1,
            help="Controls how many document chunks are retrieved for each answer.",
        )

        st.markdown("---")

        if st.button("🆕 New chat"):
            st.session_state.messages = []
            st.session_state.chat_engine = None
            st.session_state.chat_engine_top_k = None
            st.rerun()

        with st.expander("Deployment tips", expanded=False):
            st.caption(
                "If Streamlit Cloud restarts during startup, avoid loading large models at import-time. "
                "This app loads the LLM/embeddings lazily on the first question."
            )

    # Display history
    for idx, msg in enumerate(st.session_state.messages):
        role = msg.get("role", "assistant")
        content = msg.get("content", "")
        sources = msg.get("sources", [])

        with st.chat_message(role):
            st.markdown(content)

            if role == "assistant":
                if sources:
                    st.markdown("**Sources**")
                    for src in sources:
                        label = src.get("label", "Source")
                        url = src.get("url")
                        if url:
                            st.markdown(f"- [{label}]({url})")
                        else:
                            st.markdown(f"- {label}")

                col_up, col_down, _ = st.columns([0.5, 0.5, 6])
                with col_up:
                    st.button("👍", key=f"thumbs_up_{idx}")
                with col_down:
                    st.button("👎", key=f"thumbs_down_{idx}")

    # Chat input
    user_input = st.chat_input("🗣️✨ Ask me...")
    if user_input:
        # Show user message immediately
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        # Generate assistant response
        with st.chat_message("assistant"):
            try:
                # Build engine only now (lazy), preventing health-check EOF on Streamlit Cloud
                with st.spinner("Loading models and searching sources..."):
                    chat_engine = get_or_create_engine()

                response = chat_engine.chat(user_input)
                answer_text = str(response)
                source_links = extract_source_links(response)

                st.markdown(answer_text)

                if source_links:
                    st.markdown("**Sources**")
                    for src in source_links:
                        label = src["label"]
                        url = src["url"]
                        if url:
                            st.markdown(f"- [{label}]({url})")
                        else:
                            st.markdown(f"- {label}")

                col_up, col_down, _ = st.columns([0.5, 0.5, 6])
                with col_up:
                    st.button("👍", key="thumbs_up_live")
                with col_down:
                    st.button("👎", key="thumbs_down_live")

            except Exception as e:
                # Show a friendly error and keep the app alive
                st.error(
                    "The app hit an error while generating an answer. "
                    "Check that your Streamlit Secrets include GOOGLE_API_KEY, "
                    "and that your vector/index files are available in the repo."
                )
                st.exception(e)
                return

        # Store assistant message
        st.session_state.messages.append(
            {"role": "assistant", "content": answer_text, "sources": source_links}
        )


if __name__ == "__main__":
    main()
