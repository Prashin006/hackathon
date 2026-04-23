import streamlit as st
from backend.generation import generate_answer
from backend.ingestion import ingest_document
from backend.content_filter import is_safe_query
import tempfile
import os

st.title("📄 Document Q&A")

# File upload section
st.subheader("Upload Document")
uploaded_file = st.file_uploader(
    "Upload PDF or TXT file",
    type=["pdf", "txt"]
)

if uploaded_file:
    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=os.path.splitext(uploaded_file.name)[1]
    ) as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name

    with st.spinner("Ingesting document..."):
        status = ingest_document(tmp_path)
    st.success(status)

# Query section
st.subheader("Ask a Question")
query = st.text_input("Enter your question about the document:")
use_reasoning = st.checkbox(
    "Use deep reasoning model (slower but more insightful)"
)

if st.button("Get Answer") and query:
    is_safe, message = is_safe_query(query)
    if not is_safe:
        st.error(message)
    else:
        with st.spinner("Generating answer..."):
            answer = generate_answer(query, use_reasoning=use_reasoning)
        st.subheader("Answer")
        st.write(answer)