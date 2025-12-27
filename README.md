🇩🇪 German Immigration Information Chatbot (RAG-Based)

https://rag-app-chatbot-cghbrhuectc4fsypks7vzu.streamlit.app/


Overview:

ImmiExpert is a Retrieval-Augmented Generation (RAG) chatbot that answers questions about German immigration policies using authoritative documents.

Instead of relying only on a language model’s memory, the chatbot retrieves relevant information from curated PDF sources and generates accurate, explainable responses. Users can also download the source PDFs directly from the app for transparency.

This project demonstrates end-to-end AI application development, from document ingestion to cloud deployment.

Key Features:

✅ Retrieval-Augmented Generation (RAG) architecture
✅ Semantic search over PDF documents
✅ Context-aware answers using document citations
✅ Downloadable source PDFs and source link for transparency
✅ Secure API key handling with Streamlit Secrets
✅ Cloud deployment on Streamlit
✅ Clean conversational UI with chat history
✅ Configurable retrieval depth (Top-K control)


Tech Stack:

Frontend & Deployment
Streamlit
Streamlit Cloud
LLM & RAG
Google Gemini (LLM)
LlamaIndex (RAG framework)
Embeddings & Vector Search
HuggingFace Sentence Transformers
VectorStoreIndex
Backend & Utilities
Python
NumPy
PyTorch
PDF document loaders
Security
Streamlit Secrets (no hard-coded API keys)
.env ignored in version control


└── README.md


How It Works (RAG Pipeline):

Document Ingestion:
Immigration-related PDFs are loaded and chunked.

Embedding Generation:
Text chunks are converted into vector embeddings using HuggingFace models.

Vector Storage & Retrieval:
Relevant chunks are retrieved using semantic similarity.

Answer Generation:
Google Gemini generates answers grounded in retrieved documents.

Source Transparency:
Users see sources and can download the original PDFs.


Deployment (Streamlit Cloud):

App is deployed via GitHub integration
Secrets are managed using Streamlit Secrets
Automatic redeployment on each commit
Compatible with Python 3.11+


Example Questions

What are the requirements for a German Blue Card?
What are the requirements for a German settlement permit?
Can people with an opportunity card work part-time in Germany?

Why This Project Matters:

This project demonstrates practical skills in:
Building production-ready AI applications
Applying RAG architecture to reduce hallucinations
Working with real documents and PDFs
Managing cloud deployments and secrets securely
Designing user-friendly and simple AI interfaces

Next steps:

Integration with external policy APIs or government data portals
Automated compliance checks against updated immigration laws
Role-based access for advisors, administrators, and end-users
Deployment using Docker or Kubernetes for scalable infrastructure
Observability with logging, tracing, and alerting (e.g., Prometheus)
