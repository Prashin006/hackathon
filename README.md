# AI-Powered Document Intelligence System
**TCS AI Friday Hackathon | Team: [Cisco Team]**

#### Project Template
hackathon-project/
├── .env.example                  ← environment variables template
├── requirements.txt              ← all pip installs
├── README.md                     ← pre-written template, team fills in blanks
├── Token/                        ← pre-downloaded tiktoken file 
│   └── 9dsfab32432b434324bf65ac
├── config/
│   └── settings.py               ← all model names, endpoints, constants
├── backend/
│   ├── __init__.py
│   ├── ingestion.py              ← document loading, chunking, embedding
│   ├── retrieval.py              ← ChromaDB retrieval
│   ├── generation.py             ← LLM answer generation
│   ├── sql_handler.py            ← PostgreSQL + text-to-SQL
│   ├── content_filter.py         ← blocked words handler
│   └── traditional_ml.py         ← SVM, LogReg, sklearn models
├── data/
│   ├── sample_documents/         ← your pre-prepared test data
│   └── chroma_db/                ← persisted vector store
├── app.py                        ← Streamlit main page
└── pages/
    ├── 1_Document_QA.py          ← RAG Q&A interface
    ├── 2_Data_Analysis.py        ← traditional ML + insights
    └── 3_Upload.py               ← file upload + ingestion

#### Problem Statement
[Fill in on hackathon day]

#### Solution Overview
An end-to-end document intelligence system combining:
- RAG (Retrieval Augmented Generation) for unstructured documents
- PostgreSQL for structured data with Text-to-SQL
- Traditional ML for baseline analysis
- Azure-hosted LLMs via LiteLLM gateway

#### Architecture
[Attach flow diagram]

#### Tech Stack
- LLM: GPT-4o, DeepSeek-R1 via Azure LiteLLM Gateway
- Embeddings: text-embedding-3-large
- Vector DB: ChromaDB
- Structured DB: PostgreSQL
- UI: Streamlit
- Backend: Python, LangChain, SQLAlchemy, scikit-learn

#### How to Run
1. Clone the repo
2. Copy `.env.example` to `.env` and fill in values
3. pip install -r requirements.txt
4. streamlit run app.py

#### Team
- [Prashin Parikh]      — RAG Pipeline & Vector DB
- [Ayush Thada]         — Streamlit UI
- [Sanket Sawant]       — Backend & SQL
- [Gaurav Yadav]        — UI Support
- [Sakshi Padwal]       — Documentation