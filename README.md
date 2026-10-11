<div align="center">

# 🧠 LangPilot

### RAG-Powered PDF Question Answering with LangChain & ChromaDB

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![LangChain](https://img.shields.io/badge/LangChain-🦜-1C3C3C?style=for-the-badge)](https://langchain.com)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector_Store-FF6F00?style=for-the-badge)](https://www.trychroma.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)

*A Retrieval-Augmented Generation (RAG) pipeline that turns any PDF into an intelligent Q&A system — powered by LangChain, ChromaDB vector embeddings, and Groq LLM inference.*

[How It Works](#-how-it-works) · [Features](#-features) · [Architecture](#-architecture) · [Quick Start](#-quick-start) · [Tech Stack](#-tech-stack)

</div>

---

## 🎯 Why LangPilot?

Extracting specific answers from large PDF documents is tedious and time-consuming. **LangPilot** automates this with a production-grade RAG pipeline:

1. **Extract** text from any PDF document
2. **Chunk** it into semantically meaningful segments
3. **Embed** chunks as vectors and store in ChromaDB
4. **Retrieve** relevant context at query time
5. **Generate** grounded answers via LLM — with citations from the source document

The result: a conversational chat interface where you can ask natural language questions and get accurate, context-aware answers.

---

## ✨ Features

| Category | Details |
|---|---|
| 📄 **PDF Processing** | Text extraction via PyMuPDF (`fitz`) · Handles multi-page documents |
| ✂️ **Smart Chunking** | `RecursiveCharacterTextSplitter` · 500-char chunks with 50-char overlap for context continuity |
| 🧮 **Vector Embeddings** | `sentence-transformers/all-mpnet-base-v2` · Runs locally via HuggingFace |
| 🗃️ **Vector Store** | ChromaDB with persistent disk storage · Efficient similarity search |
| 🤖 **LLM Inference** | Groq API · `llama-3.1-8b-instant` · Temperature 0.0 for factual responses |
| 💬 **Chat Interface** | Streamlit UI · Dark theme · Conversation history · Real-time responses |

---

## 🏗️ Architecture

```
┌─────────────────┐
│   PDF Document   │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────────────────────────┐
│  1. EXTRACT  ─  extract.py                      │
│     PyMuPDF (fitz) → Extracted_Text.txt          │
└────────────────────────┬────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────┐
│  2. CHUNK  ─  chunks.py                          │
│     RecursiveCharacterTextSplitter → Chunks.pkl  │
│     (size: 500 | overlap: 50)                    │
└────────────────────────┬────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────┐
│  3. EMBED  ─  embeddings.py                      │
│     HuggingFace all-mpnet-base-v2                │
│     → ChromaDB (chroma_langchain_db/)            │
└────────────────────────┬────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────┐
│  4. RETRIEVE + GENERATE  ─  app.py               │
│     ChromaDB Retriever → Groq LLM               │
│     (llama-3.1-8b-instant, temp=0.0)             │
│                    ↕                             │
│           Streamlit Chat UI                      │
└─────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/prasenjitshinde11/langpilot.git
cd langpilot

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set your Groq API key
export GROQ_API_KEY="your-groq-api-key"

# Step 1: Extract text from PDF
python extract.py

# Step 2: Chunk the extracted text
python chunks.py

# Step 3: Generate embeddings & store in ChromaDB
python embeddings.py

# Step 4: Launch the chat interface
streamlit run app.py
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|------------|
| **Language** | Python 3.10+ |
| **LLM Framework** | LangChain (`langchain-core`, `langchain-groq`, `langchain-huggingface`, `langchain-chroma`) |
| **Embedding Model** | `sentence-transformers/all-mpnet-base-v2` (HuggingFace, runs locally) |
| **Vector Database** | ChromaDB (persistent disk storage) |
| **LLM Provider** | Groq API (`llama-3.1-8b-instant`) |
| **PDF Parsing** | PyMuPDF (`fitz`) |
| **Frontend** | Streamlit (custom dark-themed UI) |

---

## 📁 Project Structure

```
langpilot/
├── app.py              # Streamlit chat UI + RAG query pipeline
├── extract.py          # PDF → plain text extraction
├── chunks.py           # Text → overlapping chunks
├── embeddings.py       # Chunks → vector embeddings → ChromaDB
├── chroma_langchain_db/  # Persistent ChromaDB storage
├── requirements.txt
└── README.md
```

---

## 🔑 Key Engineering Decisions

- **RAG over Fine-Tuning** — Retrieval-Augmented Generation provides accurate, source-grounded answers without the cost and complexity of model fine-tuning
- **Local Embeddings** — `all-mpnet-base-v2` runs on CPU, eliminating API costs for the embedding step
- **Groq for Inference** — Ultra-fast LLM inference with `llama-3.1-8b-instant` at zero temperature for deterministic, factual responses
- **Persistent ChromaDB** — Embeddings are stored on disk, so you only need to process the PDF once
- **Overlapping Chunks** — 50-character overlap preserves context across chunk boundaries for better retrieval accuracy

---

## 🧪 Concepts Demonstrated

This project demonstrates practical knowledge in:

- 🔹 **Retrieval-Augmented Generation (RAG)** pipeline design
- 🔹 **Vector embeddings** and similarity search
- 🔹 **LangChain** orchestration (chains, retrievers, prompt templates)
- 🔹 **ChromaDB** vector database management
- 🔹 **LLM integration** via API (Groq)
- 🔹 **Document processing** and text chunking strategies

---

## 📬 Contact

**Prasenjit Shinde** — CS Undergrad @ SIRT Bhopal

[![GitHub](https://img.shields.io/badge/GitHub-prasenjitshinde11-181717?style=flat-square&logo=github)](https://github.com/prasenjitshinde11)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-prasenjitshinde11-0A66C2?style=flat-square&logo=linkedin)](https://linkedin.com/in/prasenjitshinde11)

---

<div align="center">

**If you found this project useful, consider giving it a ⭐**

</div>
