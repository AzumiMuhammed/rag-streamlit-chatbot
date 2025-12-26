from llama_index.llms.google_genai import GoogleGenAI
from evaluation.evaluation_config import EVALUATION_LLM_MODEL
from ragas.embeddings import HuggingFaceEmbeddings
from ragas.llms.base import LlamaIndexLLMWrapper
from evaluation.evaluation_config import EVALUATION_EMBEDDING_MODEL_NAME
from src.config import EMBEDDING_CACHE_PATH
from src.secrets import get_google_api_key

def initialise_evaluation_llm() -> GoogleGenAI:
    api_key = get_google_api_key()
    return GoogleGenAI(
        api_key=api_key,
        model=EVALUATION_LLM_MODEL,
    )

def load_ragas_models() -> tuple[LlamaIndexLLMWrapper, HuggingFaceEmbeddings]:
    print("--- 🧠 Loading Ragas LLM and Embeddings ---")
    llm_for_evaluation: GoogleGenAI = initialise_evaluation_llm()
    ragas_llm = LlamaIndexLLMWrapper(llm=llm_for_evaluation)

    ragas_embeddings = HuggingFaceEmbeddings(
        model=EVALUATION_EMBEDDING_MODEL_NAME,
        cache_folder=EMBEDDING_CACHE_PATH.as_posix(),
    )
    return ragas_llm, ragas_embeddings