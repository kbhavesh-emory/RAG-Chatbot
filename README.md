# RAG Chatbot (Local) — Streamlit + FAISS + Ollama

A GitHub-ready, fully local Retrieval-Augmented Generation (RAG) chatbot.

- Chat UI: Streamlit
- Vector search: FAISS (saved on disk)
- Embeddings: Sentence-Transformers (HuggingFace)
- LLM: Ollama (local)

No cloud keys required. Your documents stay on your machine/server.

## What this repo does

This project uses an **offline indexing step** and a **chat-only UI**:

1) Put documents in `data/uploads/`
2) Run the indexer to build a FAISS index in `data/faiss_index/`
3) Start the Streamlit chat and ask questions grounded in your documents

## Supported file types

- PDF (`.pdf`)
- Text (`.txt`)
- Word (`.docx`)
- Excel (`.xlsx`, `.xls`)
- Legacy Word (`.doc`) is best-effort (recommended: convert to `.docx`)

## Quick start (Linux/macOS)

### 1) Create a virtualenv and install deps

```bash
python3 -m venv myenv
source myenv/bin/activate
pip install -r requirements.txt
```

### 2) Install Ollama + pull a model

Install Ollama: https://ollama.com

```bash
ollama pull mistral:latest
```

Ollama should be reachable at `http://localhost:11434`.

### 3) Add documents and build the index

```bash
mkdir -p data/uploads
# copy your files into data/uploads
./myenv/bin/python embed_uploads.py
```

This creates:

- `data/faiss_index/index.faiss`
- `data/faiss_index/index.pkl`
- `data/faiss_index/manifest.json`

### 4) Start the chat UI

```bash
./start.sh
```

Open: `http://localhost:8501`

Remote server access:

```bash
HOST=0.0.0.0 PORT=8501 ./start.sh
```

## Quick start (Windows)

```powershell
python -m venv myenv
myenv\Scripts\Activate.ps1
pip install -r requirements.txt
ollama pull mistral:latest
python embed_uploads.py
streamlit run app.py
```

## How to use

- The chat UI reads the saved index from `data/faiss_index/`.
- If you add/remove documents, re-run `embed_uploads.py` and click **Reload index** in the UI.
- For best performance, click **Warm up index** after opening the page.

## Configuration (env vars)

### Chat app (`app.py`)

- `OLLAMA_BASE_URL` (default: `http://localhost:11434`)
- `OLLAMA_MODEL` (default: `mistral:latest`)
- `INDEX_DIR` (default: `data/faiss_index`)
- `K_DOCUMENTS` (default: `3`)
- `SHOW_SOURCES=1` to show retrieved chunks

### Indexer (`embed_uploads.py`)

- `UPLOADS_DIR` (default: `data/uploads`)
- `INDEX_DIR` (default: `data/faiss_index`)
- `EMBEDDING_MODEL` (default depends on script; recommended: `all-MiniLM-L6-v2` for speed)
- `MAX_CHUNKS` to cap chunk count for huge corpora

## Repo layout

- `app.py` — chat-only Streamlit UI
- `embed_uploads.py` — offline ingestion/index builder
- `rag_common.py` — shared loaders + splitting utilities
- `start.sh` / `stop.sh` — run Streamlit as a service (PID + logs)
- `data/uploads/` — your input documents
- `data/faiss_index/` — saved FAISS index

## Troubleshooting

### UI says Ollama not reachable

- Start Ollama and verify `curl http://localhost:11434` works on the same machine.
- If Ollama runs on another host, set `OLLAMA_BASE_URL`.

### Indexing is killed / too slow

- Use a smaller embedding model: `EMBEDDING_MODEL=all-MiniLM-L6-v2`
- Cap work: `MAX_CHUNKS=5000 ./myenv/bin/python embed_uploads.py`

### First message is slow

- Click **Warm up index** once, then chat.

## Security notes

- If you bind `HOST=0.0.0.0`, your Streamlit app is reachable on the network.
- Only disable CORS/XSRF (`DISABLE_CORS=1`, `DISABLE_XSRF=1`) if you are behind a trusted reverse proxy or on a private network.
