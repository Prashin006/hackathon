import os
from langchain_community.document_loaders import (
    TextLoader, PyPDFLoader, CSVLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from config.settings import (
    LITELLM_BASE_URL, LITELLM_API_KEY,
    EMBEDDING_MODEL, CHROMA_DIR,
    CHROMA_COLLECTION_NAME, CHUNK_SIZE, CHUNK_OVERLAP
)


def get_embeddings():
    """Initialize embedding model via LiteLLM gateway."""
    return OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        openai_api_base=LITELLM_BASE_URL,
        openai_api_key=LITELLM_API_KEY
    )


def load_document(file_path: str):
    """Load document based on file type."""
    ext = os.path.splitext(file_path)[1].lower()
    if ext == ".pdf":
        loader = PyPDFLoader(file_path)
    elif ext == ".txt":
        loader = TextLoader(file_path)
    elif ext == ".csv":
        loader = CSVLoader(file_path)
    else:
        raise ValueError(f"Unsupported file type: {ext}")
    return loader.load()


def ingest_document(file_path: str) -> str:
    """
    Full ingestion pipeline:
    Load → Split → Embed → Store in ChromaDB
    Returns a status message.
    """
    try:
        # Step 1: Load
        docs = load_document(file_path)

        # Step 2: Split
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP
        )
        chunks = splitter.split_documents(docs)

        # Step 3: Embed + Store
        embeddings = get_embeddings()
        Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=CHROMA_DIR,
            collection_name=CHROMA_COLLECTION_NAME
        )

        return f"✅ Successfully ingested {len(chunks)} chunks from {os.path.basename(file_path)}"

    except Exception as e:
        return f"❌ Ingestion failed: {str(e)}"