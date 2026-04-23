import streamlit as st

st.set_page_config(
    page_title="AI Hackathon — Team Solution",
    page_icon="🧠",
    layout="wide"
)

st.title("🧠 AI-Powered Document Intelligence System")
st.markdown("""
Welcome to our hackathon solution. Use the sidebar to navigate between features:

- **Document Q&A** — Upload documents and ask intelligent questions
- **Data Analysis** — Upload CSV/Excel for traditional ML insights
- **File Upload** — Manage your data sources
""")

st.info("👈 Select a page from the sidebar to get started.")