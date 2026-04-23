import pandas as pd
from sqlalchemy import create_engine, text, inspect
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from config.settings import (
    LITELLM_BASE_URL, LITELLM_API_KEY,
    PRIMARY_LLM, MAX_SQL_RETRIES
)
import os

# PostgreSQL engine
POSTGRES_URL = os.getenv("POSTGRES_URL")
engine = create_engine(POSTGRES_URL) if POSTGRES_URL else None


def load_csv_to_postgres(file_path: str, table_name: str) -> str:
    """Load a CSV or Excel file into PostgreSQL."""
    try:
        if file_path.endswith(".csv"):
            df = pd.read_csv(file_path)
        else:
            df = pd.read_excel(file_path)

        df.to_sql(table_name, engine, if_exists="replace", index=False)
        return f"✅ Loaded {len(df)} rows into table '{table_name}'"
    except Exception as e:
        return f"❌ Failed to load data: {str(e)}"


def get_schema(table_name: str) -> str:
    """Get table schema as a string for prompt injection."""
    try:
        inspector = inspect(engine)
        columns = inspector.get_columns(table_name)
        return ", ".join(
            [f"{col['name']} ({col['type']})" for col in columns]
        )
    except Exception as e:
        return f"Schema unavailable: {str(e)}"


def query_with_sql(user_question: str, table_name: str) -> str:
    """
    Self-correcting Text-to-SQL pipeline.
    LLM writes SQL → execute → if error → LLM fixes → retry.
    """
    if not engine:
        return "❌ PostgreSQL not configured."

    schema = get_schema(table_name)
    llm = ChatOpenAI(
        model=PRIMARY_LLM,
        openai_api_base=LITELLM_BASE_URL,
        openai_api_key=LITELLM_API_KEY,
        temperature=0
    )

    error_context = ""
    for attempt in range(MAX_SQL_RETRIES):
        fix_hint = (
            f"\nPrevious query failed with: {error_context}\nPlease fix it."
            if error_context else ""
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", f"""You are a SQL expert.
Table: {table_name}
Schema: {schema}
Write a valid PostgreSQL query to answer the question.
Return ONLY the raw SQL query — nothing else.{fix_hint}"""),
            ("human", "{question}")
        ])

        chain = prompt | llm | StrOutputParser()
        sql_query = chain.invoke({"question": user_question}).strip()

        try:
            with engine.connect() as conn:
                result = conn.execute(text(sql_query))
                rows = result.fetchall()
                columns = list(result.keys())

            if not rows:
                return "No results found for your query."

            # Format as readable text
            df_result = pd.DataFrame(rows, columns=columns)
            return df_result.to_string(index=False)

        except Exception as e:
            error_context = str(e)
            continue

    return f"❌ Could not execute query after {MAX_SQL_RETRIES} attempts."