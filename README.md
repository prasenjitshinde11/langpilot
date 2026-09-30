# LangPilot

A RAG (Retrieval-Augmented Generation) pipeline for PDF document Q&A.
Upload any PDF and ask questions — powered by LangChain and ChromaDB.

## What it does
- Extracts text from PDF documents
- Chunks text into segments for efficient retrieval
- Generates vector embeddings using LangChain
- Stores embeddings locally in ChromaDB
- Answers natural language questions using retrieved context + LLM

## Tech Stack
- **Python**
- **LangChain** — orchestration and retrieval chain
- **ChromaDB** — local vector store
- **PDF processing** — PyMuPDF / pdfplumber

## How to run

```bash
pip install -r requirements.txt
python app.py
```

## Architecture

```
PDF Input
   |
   v
extract.py       <- PDF text extraction
   |
   v
chunks.py        <- Text chunking
   |
   v
embeddings.py    <- Vector embedding generation (LangChain)
   |
   v
ChromaDB         <- Local vector store
   |
   v
app.py           <- RAG query chain + main interface
   |
   v
Answer Output
```

## Project Structure

| File | Purpose |
|---|---|
| `app.py` | Main RAG query chain and user interface |
| `extract.py` | PDF text extraction |
| `chunks.py` | Text chunking logic |
| `embeddings.py` | Vector embedding generation |
| `requirements.txt` | Python dependencies |
