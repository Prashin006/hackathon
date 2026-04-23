from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from backend.retrieval import retrieve_context
from backend.content_filter import is_safe_query, safe_llm_call
from config.settings import (
    LITELLM_BASE_URL, LITELLM_API_KEY,
    PRIMARY_LLM, REASONING_LLM
)


def get_llm(model: str, temperature: float = 0):
    """Initialize LLM via LiteLLM gateway."""
    return ChatOpenAI(
        model=model,
        openai_api_base=LITELLM_BASE_URL,
        openai_api_key=LITELLM_API_KEY,
        temperature=temperature
    )


def generate_answer(query: str, use_reasoning: bool = False) -> str:
    """
    Full RAG pipeline:
    Content filter → Retrieve → Generate insight
    """
    # Step 1: Content filter
    is_safe, message = is_safe_query(query)
    if not is_safe:
        return message

    # Step 2: Retrieve relevant context
    context = retrieve_context(query)

    # Step 3: Generate answer
    model = REASONING_LLM if use_reasoning else PRIMARY_LLM
    llm = get_llm(model)

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an expert analyst.
Use the provided context to answer the user's query.
Always provide a clear, narrative, insight-driven answer.
Do not use bullet points — write in flowing paragraphs.
If the context does not contain enough information, say so clearly.
Never hallucinate or make up facts.

Context:
{context}
"""),
        ("human", "{query}")
    ])

    chain = prompt | llm | StrOutputParser()

    def llm_call():
        return chain.invoke({"context": context, "query": query})

    return safe_llm_call(llm_call)