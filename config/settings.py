import os
from dotenv import load_dotenv

load_dotenv()

# Tiktoken cache — CRITICAL for Azure SSL fix
os.environ["TIKTOKEN_CACHE_DIR"] = os.getenv(
    "TIKTOKEN_CACHE_DIR", "./tiktoken_cache"
)

# LiteLLM Gateway
LITELLM_BASE_URL = os.getenv("LITELLM_BASE_URL")
LITELLM_API_KEY  = os.getenv("LITELLM_API_KEY")

# Model names
PRIMARY_LLM      = os.getenv("PRIMARY_LLM",      "azure/genailab-maas-gpt-4o")
FAST_LLM         = os.getenv("FAST_LLM",          "azure/genailab-maas-gpt-4o-mini")
EMBEDDING_MODEL  = os.getenv("EMBEDDING_MODEL",   "azure/genailab-maas-text-embedding-3-large")
REASONING_LLM    = os.getenv("REASONING_LLM",     "azure/genailab-maas-DeepSeek-R1")

# ChromaDB
CHROMA_DIR             = "./data/chroma_db"
CHROMA_COLLECTION_NAME = "hackathon_docs"

# Chunking
CHUNK_SIZE    = 500
CHUNK_OVERLAP = 50

# Self-correction
MAX_SQL_RETRIES = 3
MAX_RAG_RETRIES = 2