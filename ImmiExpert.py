import streamlit as st

from src.model_loader import initialise_llm, get_embedding_model
from src.engine import get_chat_engine

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.google_genai import GoogleGenAI  # adjust path if needed


# ---------------------------------------------------------
# Basic page config
# ---------------------------------------------------------
st.set_page_config(
    page_title="ImmiExpert",
    page_icon="🤖✨",
    layout="wide",
)


# ---------------------------------------------------------
# Source → URL mapping (for your PDFs / docs)
# ---------------------------------------------------------
# We’ll try to match substrings in the file name to these keys.
SOURCE_URL_MAP: dict[str, tuple[str, str]] = {
     #key in filename        (label,                                                                       url)
    "entry": (
        "Entry regulations for Germany",
        "https://www.bamf.de/EN/Startseite/startseite_node.html",
    ),
    "BAMF_Bundesamt für Migration und Flüchtlinge _Entry regulations": (
        "BAMF_Bundesamt für Migration und Flüchtlinge _Entry regulations",
        "https://www.bamf.de/EN/Startseite/startseite_node.html",
    ),
    "BMWK_Fast-track_procedure_for_skilled_workers_2024_EN-bf":(
        "BMWK_Fast-track_procedure_for_skilled_workers_2024_EN-bf",
        "https://www.make-it-in-germany.com/en/looking-for-foreign-professionals/entering/the-fast-track-procedure-for-skilled-workers",
    ),
    "housing_registration": (
        "housing_registration",
        "https://www.make-it-in-germany.com/en/",
    ),    
    
    "Study-Visa": (
        "Study-Visa",
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
        "Settlement-permit",
        "https://www.make-it-in-germany.com/en/visa-residence/living-permanently/settlement-permit",
    ),
    "working": (
        "Working in Germany",
        "https://www.tatsachen-ueber-deutschland.de/en",
    ),

    "German Citizenship": (
        "German Citizenship",
        "https://www.tatsachen-ueber-deutschland.de/en/migration-and-integration/more-inclusion-thanks-citizenship",
    ),
}


def map_source_to_url(file_name: str) -> tuple[str, str | None]:
    """
    Map a file name like 'settlement-permit.pdf' to a
    human-readable label and external URL (if known).
    """
    lower_name = file_name.lower()

    for key, (label, url) in SOURCE_URL_MAP.items():
        if key in lower_name:
            return label, url

    # Fallback: show just the file name without link
    short_name = file_name.split("/")[-1]
    return short_name, None


def extract_source_links(response) -> list[dict]:
    """
    Extract unique sources from a LlamaIndex Response object and
    convert them into a list of {label, url, raw} dicts.
    """
    source_links: list[dict] = []
    seen: set[str] = set()

    # response.source_nodes is typical for LlamaIndex Responses
    for node in getattr(response, "source_nodes", []):
        meta = getattr(node, "metadata", {}) or {}
        file_name = (
            meta.get("file_name")
            or meta.get("filename")
            or meta.get("source")
            or meta.get("doc_id")
        )

        if not file_name:
            continue

        if file_name in seen:
            continue

        seen.add(file_name)
        label, url = map_source_to_url(file_name)
        source_links.append(
            {
                "label": label,
                "url": url,
                "raw": file_name,
            }
        )

    return source_links


# ---------------------------------------------------------
# Chat engine initialisation
# ---------------------------------------------------------

#@st.cache_resource
def init_chat_engine() -> None:
    """Initialise the chat engine once and store it in session_state."""
    if "chat_engine" not in st.session_state:
        llm: GoogleGenAI = initialise_llm()
        embed_model: HuggingFaceEmbedding = get_embedding_model()
        st.session_state.chat_engine = get_chat_engine(llm, embed_model)
        st.session_state.messages = []  # list of dicts: {role, content, sources?}

    # default UI config state
    if "similarity_top_k" not in st.session_state:
        st.session_state.similarity_top_k = 4  # logical default for slider


# ---------------------------------------------------------
# Main app
# ---------------------------------------------------------
def main() -> None:
    # --- Top area (title + caption) with light-red accent ---
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

    init_chat_engine()

    # -----------------------------------------------------
    # Sidebar – session controls + simple config panel
    # -----------------------------------------------------
    with st.sidebar:
        st.markdown("### ⚙️ Settings")

        # Slider for similarity_top_k – this is stored in session_state
        similarity_top_k = st.slider(
            "Number of chunks to retrieve",
            min_value=2,
            max_value=10,
            value=st.session_state.similarity_top_k,
            step=1,
            help="Controls how many document chunks are retrieved for each answer.",
        )
        st.session_state.similarity_top_k = similarity_top_k

#        st.info(
#            "ℹ️ To fully apply this slider, update your "
#            "`get_chat_engine` in `src.engine` to use "
#            "`similarity_top_k=st.session_state.similarity_top_k` "
#            "when calling `as_chat_engine()`."
#        )

        st.markdown("---")
#       st.markdown("### 💬 Session controls")

        if st.button("🆕 New chat"):
            # Clear stored messages
            st.session_state.messages = []

            # Reset chat engine (fresh memory)
            llm = initialise_llm()
            embed_model = get_embedding_model()
            st.session_state.chat_engine = get_chat_engine(llm, embed_model)

            # Rerun to refresh UI
            st.rerun()

    # -----------------------------------------------------
    # Main chat area – display history
    # -----------------------------------------------------
    if "messages" in st.session_state:
        for idx, msg in enumerate(st.session_state.messages):
            role = msg.get("role", "assistant")
            content = msg.get("content", "")
            sources = msg.get("sources", [])

            with st.chat_message(role):
                st.markdown(content)

                # If this is an assistant message, show sources + feedback
                if role == "assistant":
                    if sources:
                        st.markdown("**Sources**")
                        for src in sources:
                            label = src["label"]
                            url = src["url"]
                            if url:
                                st.markdown(f"- [{label}]({url})")
                            else:
                                st.markdown(f"- {label}")

                    # Thumbs up / down (non-functional for now)
                    col_up, col_down, _ = st.columns([0.5, 0.5, 6])
                    with col_up:
                        st.button("👍", key=f"thumbs_up_{idx}")
                    with col_down:
                        st.button("👎", key=f"thumbs_down_{idx}")

    # -----------------------------------------------------
    # Chat input at the bottom
    # -----------------------------------------------------
    user_input = st.chat_input("🗣️✨ Ask me...")
    if user_input:
        # 1. Show user message immediately
        st.session_state.messages.append(
            {"role": "user", "content": user_input}
        )
        with st.chat_message("user"):
            st.markdown(user_input)

        # 2. Get response from RAG chat engine
        with st.chat_message("assistant"):
            chat_engine = st.session_state.chat_engine
            response = chat_engine.chat(user_input)
            answer_text = str(response)

            # Extract sources from the response
            source_links = extract_source_links(response)

            # Show answer text
            st.markdown(answer_text)

            # Show sources below the answer
            if source_links:
                st.markdown("**Sources**")
                for src in source_links:
                    label = src["label"]
                    url = src["url"]
                    if url:
                        st.markdown(f"- [{label}]({url})")
                    else:
                        st.markdown(f"- {label}")

            # Feedback buttons (non-functional for now)
            col_up, col_down, _ = st.columns([0.5, 0.5, 6])
            with col_up:
                st.button("👍", key="thumbs_up_live")
            with col_down:
                st.button("👎", key="thumbs_down_live")

        # 3. Store assistant message + its sources in history
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": answer_text,
                "sources": source_links,
            }
        )


if __name__ == "__main__":
    main()