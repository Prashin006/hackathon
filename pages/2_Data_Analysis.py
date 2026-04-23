import streamlit as st
import pandas as pd
from backend.traditional_ml import run_classification, run_regression
from backend.sql_handler import load_csv_to_postgres, query_with_sql
from backend.generation import generate_answer

st.title("📊 Data Analysis")

uploaded_file = st.file_uploader(
    "Upload CSV or Excel file",
    type=["csv", "xlsx"]
)

if uploaded_file:
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    st.subheader("Data Preview")
    st.dataframe(df.head())

    # Store in PostgreSQL
    table_name = st.text_input("Table name for PostgreSQL:", "uploaded_data")
    if st.button("Store in Database"):
        import tempfile, os
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=os.path.splitext(uploaded_file.name)[1]
        ) as tmp:
            tmp.write(uploaded_file.getvalue())
            status = load_csv_to_postgres(tmp.name, table_name)
        st.success(status)

    st.divider()

    # Traditional ML
    st.subheader("Traditional ML Analysis")
    target_col = st.selectbox("Select target column:", df.columns.tolist())
    task_type = st.radio("Task type:", ["Classification", "Regression"])

    if st.button("Run ML Analysis"):
        with st.spinner("Running analysis..."):
            if task_type == "Classification":
                results = run_classification(df, target_col)
            else:
                results = run_regression(df, target_col)

        for model_name, metrics in results.items():
            st.write(f"**{model_name}**")
            st.json(metrics)

    st.divider()

    # Natural language SQL query
    st.subheader("Ask Questions About Your Data")
    nl_query = st.text_input("Ask a question about your data:")
    if st.button("Query Data") and nl_query:
        with st.spinner("Querying..."):
            result = query_with_sql(nl_query, table_name)
        st.text(result)