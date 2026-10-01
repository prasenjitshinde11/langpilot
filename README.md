# LangPilot

A RAG (Retrieval-Augmented Generation) pipeline that answers questions about PDF documents using LangChain, ChromaDB, and Groq.

## Problem

Reading and extracting specific information from large PDF documents is time-consuming. Searching through them manually is inefficient, especially when you need precise answers to targeted questions.

## Solution

LangPilot implements a RAG pipeline that:
1. Extracts text from a PDF document
2. Splits the text into overlapping chunks
3. Generates vector embeddings and stores them in ChromaDB
4. At query time, retrieves the most relevant chunks and passes them to a Groq-hosted LLM for a grounded answer

The result is a Streamlit chat interface where you can ask natural language questions about the loaded PDF.

## Features

- PDF text extraction using PyMuPDF (`fitz`)
- Text chunking with `RecursiveCharacterTextSplitter` (chunk size: 500, overlap: 50)
- Vector embeddings via `sentence-transformers/all-mpnet-base-v2` (HuggingFace)
- Persistent vector store with ChromaDB (stored locally in `chroma_langchain_db/`)
- Chat interface built with Streamlit — dark-themed, custom CSS
- LLM inference via Groq API (`llama-3.1-8b-instant`, temperature 0.0)
- Conversation history maintained in Streamlit session state

## Architecture

```
PDF file
   ↓
extract.py  →  Extracted_Text.txt   (PyMuPDF)
   ↓
chunks.py   →  Chunks.pkl           (LangChain RecursiveCharacterTextSplitter)
   ↓
embeddings.py → chroma_langchain_db/ (HuggingFace Embeddings + ChromaDB)
   ↓
app.py      →  Streamlit Chat UI    (ChromaDB retriever + Groq LLM)
```

**Key components:**
- **Embedding model:** `sentence-transformers/all-mpnet-base-v2` (runs locally)
- **LLM:** `llama-3.1-8b-instant` via Groq API
- **Vector store:** ChromaDB (persisted to disk)
- **Frontend:** Streamlit with custom dark UI

## Tech Stack

**AI / LLM**
- LangChain (`langchain-core`, `langchain-groq`, `langchain-huggingface`, `langchain-chroma`)
- Groq API — `llama-3.1-8b-instant`
- HuggingFace Sentence Transformers — `all-mpnet-base-v2`
- ChromaDB

**Backend / Processing**
- Python
- PyMuPDF (`fitz`) — PDF text extraction
- `pickle` — chunk serialization

**Frontend**
- Streamlit

## Project Structure

```
LangPilot/
├── extract.py          # Step 1: Extract text from PDF → Extracted_Text.txt
├── chunks.py           # Step 2: Split text into chunks → Chunks.pkl
├── embeddings.py       # Step 3: Embed chunks → chroma_langchain_db/
├── app.py              # Step 4: Streamlit chat UI (RAG query loop)
├── basic.py            # Standalone script — loads model + vector store
├── requirements.txt
└── chroma_langchain_db/ # Persisted ChromaDB vector store
```

## Getting Started

### Prerequisites

- Python 3.10+
- A [Groq API key](https://console.groq.com/)

### Installation

```bash
git clone https://github.com/prasenjitshinde11/LangPilot.git
cd LangPilot
pip install -r requirements.txt
```

## Environment Variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
```

> **Note:** Never commit your real API key. The `.env` file is loaded via `python-dotenv`.

## Running Locally

The pipeline runs in 4 sequential steps. You must run steps 1–3 once before launching the app.

**Step 1 — Extract text from your PDF:**
```bash
# Edit extract.py to point to your PDF file, then:
python extract.py
```
This produces `Extracted_Text.txt`.

**Step 2 — Chunk the text:**
```bash
python chunks.py
```
This produces `Chunks.pkl`.

**Step 3 — Generate embeddings and persist to ChromaDB:**
```bash
python embeddings.py
```
This populates `chroma_langchain_db/`.

**Step 4 — Launch the chat app:**
```bash
streamlit run app.py
```
Open `http://localhost:8501` in your browser.

## Testing

No automated tests are currently implemented.

## Deployment

No deployment configuration is currently included. The app runs locally via Streamlit.

## Future Improvements

- Support uploading PDFs directly through the Streamlit UI
- Support multiple PDF files and multi-document retrieval
- Add source citation — show which chunk was used to answer each question
- Add a `.gitignore` to exclude `Chunks.pkl`, `chroma_langchain_db/`, `.vscode/`, and sample files
- Add a `requirements.txt` with pinned versions for reproducibility
- Dockerize the app for easier setup

## Author

**Prasenjit Shinde**
