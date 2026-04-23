from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from config.settings import (
    LITELLM_BASE_URL, LITELLM_API_KEY,
    EMBEDDING_MODEL, CHROMA_DIR,
    CHROMA_COLLECTION_NAME
)


def get_retriever(k: int = 4):
    """Load ChromaDB and return a retriever."""
    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL,
        openai_api_base=LITELLM_BASE_URL,
        openai_api_key=LITELLM_API_KEY
    )
    vector_db = Chroma(
        persist_directory=CHROMA_DIR,
        embedding_function=embeddings,
        collection_name=CHROMA_COLLECTION_NAME
    )
    return vector_db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": k}
    )


def retrieve_context(query: str, k: int = 4) -> str:
    """Retrieve top K relevant chunks for a query."""
    retriever = get_retriever(k=k)
    docs = retriever.invoke(query)
    return "\n\n".join([doc.page_content for doc in docs])