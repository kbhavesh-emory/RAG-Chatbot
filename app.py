"""Chat-only RAG UI (ChatGPT-style).

This UI intentionally hides configuration, document upload and embedding steps.
Indexing should be done offline via embed_uploads.py which writes a FAISS index
to data/faiss_index.
"""

from __future__ import annotations

import os

# Ensure Hugging Face + PyTorch stay CPU-only on servers with incompatible CUDA.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import json
from typing import Any, Dict, List

import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.llms import Ollama
from langchain_community.vectorstores import FAISS


INDEX_DIR = os.getenv("INDEX_DIR", "data/faiss_index")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral:latest")
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.3"))

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-mpnet-base-v2")


def index_files_present(index_dir: str) -> bool:
    return bool(
        os.path.isdir(index_dir)
        and os.path.exists(os.path.join(index_dir, "index.faiss"))
        and os.path.exists(os.path.join(index_dir, "index.pkl"))
    )


def read_index_manifest(index_dir: str) -> Dict[str, Any] | None:
    manifest_path = os.path.join(index_dir, "manifest.json")
    if not os.path.exists(manifest_path):
        return None
    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

K_DOCUMENTS = int(os.getenv("K_DOCUMENTS", "3"))

SHOW_SOURCES = os.getenv("SHOW_SOURCES", "0") == "1"


st.set_page_config(
    page_title="Chat",
    page_icon="💬",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown(
    """
<style>
  /* Make the app feel more like a chat product */
  .block-container { max-width: 900px; padding-top: 2rem; }
  header { visibility: hidden; }
  footer { visibility: hidden; }
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_resource
def load_embeddings(model_name: str):
    return HuggingFaceEmbeddings(model_name=f"sentence-transformers/{model_name}")


@st.cache_resource
def load_llm(model_name: str, temperature: float, base_url: str):
    return Ollama(model=model_name, temperature=temperature, base_url=base_url)


def load_vector_store(index_dir: str, embeddings):
    if not os.path.isdir(index_dir):
        return None
    if not (
        os.path.exists(os.path.join(index_dir, "index.faiss"))
        and os.path.exists(os.path.join(index_dir, "index.pkl"))
    ):
        return None

    try:
        return FAISS.load_local(index_dir, embeddings, allow_dangerous_deserialization=True)
    except TypeError:
        return FAISS.load_local(index_dir, embeddings)


def get_vector_store_cached(index_dir: str, embedding_model_name: str):
    """Load the vector store once per Streamlit session.

    Streamlit reruns the script on every interaction; without caching, loading
    FAISS + initializing embeddings can make every message feel slow.
    """

    cache_key = f"{index_dir}::{embedding_model_name}"
    if st.session_state.get("_vector_store_key") == cache_key:
        cached = st.session_state.get("_vector_store")
        if cached is not None:
            return cached

    embeddings = load_embeddings(embedding_model_name)
    vector_store = load_vector_store(index_dir, embeddings)

    st.session_state["_vector_store_key"] = cache_key
    st.session_state["_vector_store"] = vector_store
    return vector_store


def ollama_is_running() -> bool:
    try:
        test = Ollama(model=OLLAMA_MODEL, base_url=OLLAMA_BASE_URL)
        try:
            test.invoke("ping")
            return True
        except Exception:
            try:
                test("ping")
                return True
            except Exception:
                return False
    except Exception:
        return False


def answer_question(vector_store, llm, question: str, k: int) -> Dict[str, Any]:
    docs = vector_store.similarity_search(question, k=k)
    context = "\n\n".join([d.page_content for d in docs])

    prompt = (
        "You are a helpful assistant. Answer using ONLY the provided context. "
        'If the answer is not in the context, say "I don\'t have enough information to answer this question."\n\n'
        f"Context:\n{context}\n\n"
        f"User question: {question}\n\n"
        "Answer:"
    )

    try:
        text = llm.invoke(prompt)
    except AttributeError:
        text = llm(prompt)

    return {"result": text, "source_documents": docs}


if "messages" not in st.session_state:
    st.session_state.messages = []


manifest = read_index_manifest(INDEX_DIR)
effective_embedding_model = (
    str(manifest.get("embedding_model")) if isinstance(manifest, dict) and manifest.get("embedding_model") else EMBEDDING_MODEL
)

top_left, top_right = st.columns([1, 1])
with top_left:
    st.markdown("## Chat")
    st.caption(f"Model: {OLLAMA_MODEL} · Index: {INDEX_DIR} · Embeddings: {effective_embedding_model}")
with top_right:
    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


if not ollama_is_running():
    st.error(
        "Ollama is not reachable. Start it on the server and make sure it listens on "
        f"{OLLAMA_BASE_URL}."
    )
    st.stop()


index_ready = index_files_present(INDEX_DIR)
if not index_ready:
    st.warning("No saved index found yet — answers will be limited until you build it.")
    st.code(
        "# 1) Put files here\n"
        "ls -lh data/uploads\n\n"
        "# 2) Build index\n"
        "./myenv/bin/python embed_uploads.py\n\n"
        "# 3) Reload this page (or click Reload index)\n",
        language="bash",
    )

    if st.button("Reload index", use_container_width=True):
        st.cache_resource.clear()
        st.session_state.pop("_vector_store", None)
        st.session_state.pop("_vector_store_key", None)
        st.rerun()

else:
    cols = st.columns([1, 1, 2])
    with cols[0]:
        if st.button("Warm up index", use_container_width=True):
            with st.spinner("Loading index into memory..."):
                _ = get_vector_store_cached(INDEX_DIR, effective_embedding_model)
            st.success("Index loaded for this session.")
    with cols[1]:
        if st.button("Reload index", use_container_width=True):
            st.cache_resource.clear()
            st.session_state.pop("_vector_store", None)
            st.session_state.pop("_vector_store_key", None)
            st.rerun()


for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if SHOW_SOURCES and msg.get("sources"):
            with st.expander("Sources"):
                for i, src in enumerate(msg["sources"], 1):
                    st.markdown(f"**Source {i}:**")
                    st.text(src)


user_input = st.chat_input("Message")
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})

    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        if not ollama_is_running():
            answer = (
                "Ollama is not reachable from this server.\n\n"
                f"Start it and ensure it listens on {OLLAMA_BASE_URL} (and the model `{OLLAMA_MODEL}` is available)."
            )
            st.error(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
        elif not index_ready:
            answer = (
                "I don't have a document index yet.\n\n"
                "Build it by running `./myenv/bin/python embed_uploads.py` after placing files in `data/uploads/`, "
                "then click **Reload index**."
            )
            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})
        else:
            with st.spinner("Loading index and thinking..."):
                try:
                    llm = load_llm(OLLAMA_MODEL, TEMPERATURE, OLLAMA_BASE_URL)
                    vector_store = get_vector_store_cached(INDEX_DIR, effective_embedding_model)
                    if vector_store is None:
                        raise RuntimeError(
                            "Index files exist but could not be loaded. Try clicking 'Reload index' or rebuild the index."
                        )

                    resp = answer_question(vector_store, llm, user_input, k=K_DOCUMENTS)
                    answer = resp["result"]
                    st.markdown(answer)

                    sources_preview: List[str] = []
                    if SHOW_SOURCES:
                        for d in resp.get("source_documents", [])[:K_DOCUMENTS]:
                            sources_preview.append((d.page_content or "")[:800])

                    st.session_state.messages.append(
                        {
                            "role": "assistant",
                            "content": answer,
                            "sources": sources_preview,
                        }
                    )
                except Exception as e:
                    st.error(str(e))
