# --- LLM Model Configuration ---

#LLM_MODEL: str = "gemini-2.5-flash"
#LLM_MAX_NEW_TOKENS: int = 768
#LLM_TEMPERATURE: float = 0.02
#LLM_TOP_P: float = 0.95
#LLM_REPETITION_PENALTY: float = 1.03
#LLM_QUESTION: str = "Which language is this: print('hello world!')"


# --- Conversational Chartbot Configuration

from pathlib import Path

LLM_MODEL: str = "gemini-2.5-flash"
LLM_MAX_NEW_TOKENS: int = 2000
LLM_TEMPERATURE: float = 0.02
LLM_TOP_P: float = 0.95
LLM_REPETITION_PENALTY: float = 1.03
LLM_SYSTEM_PROMPT: str = (
    "You are a helpful chatbot. Be friendly and conversational and do not use words like according to the documents or information that I have."
)

#from pathlib import Path
# --- Embedding Model Configuration ---

# EMBEDDING_MODEL_NAME: str = "sentence-transformers/all-MiniLM-L6-v2"

EMBEDDING_MODEL_NAME: str = "intfloat/multilingual-e5-small" # Multilingual embedding model


# --- RAG/VectorStore Configuration ---
# The number of most relevant text chunks to retrieve from the vector store
SIMILARITY_TOP_K: int = 4
# The size of each text chunk in tokens
CHUNK_SIZE: int = 1500
# The overlap between adjacent text chunks in tokens
CHUNK_OVERLAP: int = 100


# --- Chat Memory Configuration ---
CHAT_MEMORY_TOKEN_LIMIT: int = 6000


# --- Persistent Storage Paths (using pathlib for robust path handling) ---
ROOT_PATH: Path = Path(__file__).parent.parent
DATA_PATH: Path = ROOT_PATH / "data/"
EMBEDDING_CACHE_PATH: Path = ROOT_PATH / "local_storage/embedding_model/"
VECTOR_STORE_PATH: Path = ROOT_PATH / "local_storage/vector_store/"