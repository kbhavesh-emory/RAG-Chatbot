# RAG Chatbot using Streamlit, FAISS, Ollama, and HuggingFace Embeddings

This project implements a fully local Retrieval-Augmented Generation (RAG) chatbot using Streamlit for the UI, FAISS for vector searching, HuggingFace sentence-transformer embeddings, and Ollama as the local LLM backend.
It allows uploading PDF/Text documents, chunking and embedding them, storing the embeddings in FAISS, and answering user queries by retrieving relevant chunks and generating responses using an LLM.

The project requires no external APIs and runs completely on your local system.

<img width="950" height="434" alt="RAG Chatbot pic" src="https://github.com/user-attachments/assets/4d3389f6-e8fa-4445-accc-9503ad0505eb" />

---

## 1. Project Overview

The goal of this project is to create a small, lightweight RAG pipeline that can be run on any machine without GPU requirements or cloud services.
The app lets users upload documents, processes them into embeddings, stores them in FAISS, and queries them interactively through a chat interface.

Key capabilities:

* Upload PDFs and text files
* Extract and preprocess document content
* Split text into chunks using a custom splitter
* Generate embeddings locally using HuggingFace models
* Build a FAISS vector store for semantic search
* Query relevant chunks and use them as context for a local LLM (Ollama)
* Display answers along with source documents
* Fully local execution, meaning no data leaves your machine

---

## 2. Tech Stack

### Backend and Application Framework

* **Python**
* **Streamlit**: For the UI and session handling

### RAG Components

* **HuggingFace Sentence Transformer Embeddings** (all-MiniLM-L6-v2, all-mpnet-base-v2)
* **FAISS**: Vector store for similarity search
* **Custom text splitter**
* **Custom document class with unique ID field**

### LLM

* **Ollama Local Models**
  Supports models like:

  * mistral
  * neural-chat
  * llama2

### Document Loaders

* `PyPDFLoader` for PDF files
* `TextLoader` for text files

Everything is sourced from `langchain_community`, avoiding deprecated imports.

### Project Structure:

<img width="416" height="862" alt="image" src="https://github.com/user-attachments/assets/0a5ff573-3786-4b56-8be6-44f9464c9c56" />
---

## 3. Features

1. Local document upload (PDF or text)
2. Document preprocessing and chunking
3. Embedding creation using open-source embedding models
4. FAISS similarity search
5. Custom lightweight RAG implementation (no LangChain chains)
6. Chat-based interface for querying documents
7. Source chunk inspection
8. Configurable:

   * LLM model
   * Embedding model
   * Temperature
   * Number of documents retrieved
9. Local execution with no API keys required

---

## 4. Explanation of Each Major Code Component

### 4.1. SimpleDocument Class

A custom dataclass to wrap document content, metadata, and a unique identifier.

```python
@dataclass
class SimpleDocument:
    page_content: str
    metadata: dict
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
```

Why needed:

* FAISS requires documents to contain stable IDs for consistency.
* LangChain’s default Document class lacks this, so a custom one was created.

---

### 4.2. SimpleTextSplitter

A minimal text splitter that divides documents into chunks based on character length with overlap.

Key behaviors:

* Splits by chunk size
* Attempts to break text at natural delimiters (periods, newlines, spaces)
* Adds overlap to preserve context between chunks

Important functions:

* `split_text()`: Performs raw splitting logic
* `split_documents()`: Converts full documents into multiple chunked SimpleDocuments

<img width="949" height="440" alt="rag chatbot1" src="https://github.com/user-attachments/assets/1919d4e2-1428-4d8b-bacd-4eb89e96ab9d" />

---

### 4.3. SimpleQA Class

A small, dependency-free implementation of a RAG question-answering step.

How it works:

* Performs similarity search on the FAISS store
* Builds a context string using retrieved document chunks
* Creates a prompt manually
* Sends the prompt to the LLM using `.invoke()`, `.generate()`, or a fallback direct call
* Returns the answer and the source documents

This avoids LangChain chains entirely and provides full control.

<img width="953" height="442" alt="rag chatbot2" src="https://github.com/user-attachments/assets/6332a208-1145-4f17-ae1b-222acc6d2854" />

---

### 4.4. Streamlit Application Structure

#### Session State

Tracks:

* chat history
* vector store
* QA engine
* document load status

This prevents reloading everything on every page refresh.

---

### 4.5. Sidebar Configuration

Users can configure:

* uploaded files
* LLM model
* temperature
* embedding model
* number of retrieved chunks

This makes the app flexible and interactive.

---

### 4.6. Embedding and LLM Loaders

Cached using `@st.cache_resource`:

```python
def load_embeddings(model_name)
def load_llm(model_name, temperature)
```

Caching avoids reloading heavy models each time the app runs.

<img width="948" height="437" alt="rag chatbot3" src="https://github.com/user-attachments/assets/dc2f34e1-f33c-4078-bd26-e7ee60814813" />

---

### 4.7. Document Loading Logic

The app temporarily writes uploaded files to disk and loads them using:

* `PyPDFLoader` for PDFs
* `TextLoader` for .txt files

After loading, files are immediately deleted from disk.

Each loaded document is wrapped into a `SimpleDocument` with a unique ID.

<img width="959" height="442" alt="chatbot4" src="https://github.com/user-attachments/assets/fdbae63c-6f0e-4f39-8699-de677b94bd27" />

---

### 4.8. Vector Store Creation

Documents are chunked, embedded, and loaded into FAISS:

```python
vector_store = FAISS.from_documents(text_chunks, embeddings)
```

Chunk count is displayed so users know how much data is indexed.

---

### 4.9. Chat Interface

The chat section:

* Displays prior messages
* Provides a text input for new questions
* Shows responses generated by the RAG system
* Allows users to view the source documents via an expander

After the assistant replies, the app reruns to maintain the chat layout.

<img width="946" height="440" alt="chatbot5" src="https://github.com/user-attachments/assets/549e452d-431a-4e54-b688-2f1a76a0ff40" />

---

## 5. How the RAG Workflow Operates

1. **User uploads documents**
2. Documents are parsed into pages
3. Pages are chunked into 500-character segments with overlap
4. Chunks are embedded using HuggingFace embeddings
5. FAISS indexes all embeddings
6. User submits a query
7. The app retrieves top-k similar chunks
8. These chunks form the context for the LLM prompt
9. Ollama generates a contextual answer
10. The UI displays the answer and source chunks

This is a classic RAG pipeline implemented from scratch with minimum dependencies.

---

## 6. Possible Extensions

* Add chat history-aware prompting
* Add support for multiple local embedding models
* Enable saving/loading FAISS index
* Add multimodal RAG (PDF images + text)
* Switch to GPU embeddings if available

---

## 7. Conclusion

This project shows how a simple RAG chatbot can be built entirely using local open-source tools.
It demonstrates document parsing, custom chunking, embedding generation, FAISS indexing, and LLM-powered retrieval-based answering, all wrapped inside a clean Streamlit interface.

The entire pipeline is transparent, lightweight, and runs fully offline.
