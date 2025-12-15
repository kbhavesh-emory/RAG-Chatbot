#!/usr/bin/env python3
"""Build a FAISS vector index from everything in data/uploads.

Usage:
  ./myenv/bin/python embed_uploads.py

Common env vars:
  UPLOADS_DIR=data/uploads
  INDEX_DIR=data/faiss_index
  EMBEDDING_MODEL=all-mpnet-base-v2
  CHUNK_SIZE=500
  CHUNK_OVERLAP=50
  EXCEL_MAX_ROWS=unlimited
  EXCEL_MAX_COLS=unlimited

Notes:
  - The saved index can be loaded by app.py.
  - For legacy .doc files, install LibreOffice (soffice) or antiword.
"""

from __future__ import annotations

import os
import sys
from typing import List, Optional

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from rag_common import (
    SimpleTextSplitter,
    list_upload_files,
    load_file_to_documents,
    write_manifest,
)


def _parse_optional_limit_env(env_name: str, default_value: int | None) -> int | None:
    raw = os.getenv(env_name)
    if raw is None:
        return default_value

    value = raw.strip()
    if value == "":
        return default_value

    lowered = value.lower()
    if lowered in {"unlimited", "unlomited", "none", "null", "inf", "infinite", "no-limit", "nolimit"}:
        return None

    parsed = int(value)
    if parsed <= 0:
        return None

    return parsed


def main() -> int:
    uploads_dir = os.getenv("UPLOADS_DIR", "data/uploads")
    index_dir = os.getenv("INDEX_DIR", "data/faiss_index")

    # Default to a fast embedding model. Override with EMBEDDING_MODEL=all-mpnet-base-v2
    # if you want higher-quality embeddings.
    embedding_model_short = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

    # Defaults tuned for large PDFs (fewer chunks, less overhead).
    chunk_size = int(os.getenv("CHUNK_SIZE", "1200"))
    chunk_overlap = int(os.getenv("CHUNK_OVERLAP", "100"))

    # Controls to avoid OOM / SIGKILL (exit code 137)
    faiss_add_batch = int(os.getenv("FAISS_ADD_BATCH", "32"))
    hf_batch_size = int(os.getenv("HF_BATCH_SIZE", "8"))

    # Safety cap to prevent runaway indexing on shared systems.
    # Set MAX_CHUNKS=0 to disable.
    max_chunks = int(os.getenv("MAX_CHUNKS", "20000"))

    # Reduce thread explosions (helps avoid job-killer policies).
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    # Avoid trying to use CUDA on systems where PyTorch/CUDA arch mismatches exist.
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    try:
        import torch

        torch_threads = int(os.getenv("TORCH_NUM_THREADS", "2"))
        torch.set_num_threads(max(1, torch_threads))
        torch.set_num_interop_threads(1)
    except Exception:
        pass

    excel_max_rows = _parse_optional_limit_env("EXCEL_MAX_ROWS", 2000)
    excel_max_cols = _parse_optional_limit_env("EXCEL_MAX_COLS", 50)

    file_paths = list_upload_files(uploads_dir)
    if not file_paths:
        print(f"No supported files found in: {uploads_dir}")
        return 2

    print(f"Found {len(file_paths)} file(s) under {uploads_dir}")

    splitter = SimpleTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    # Force CPU to avoid CUDA incompatibility / extra GPU memory behavior.
    embeddings = HuggingFaceEmbeddings(
        model_name=f"sentence-transformers/{embedding_model_short}",
        model_kwargs={"device": "cpu"},
        encode_kwargs={"batch_size": hf_batch_size},
    )

    vector_store: Optional[FAISS] = None
    pending_batch = []
    chunk_count = 0

    def flush_batch() -> None:
        nonlocal vector_store, pending_batch
        if not pending_batch:
            return
        if vector_store is None:
            vector_store = FAISS.from_documents(pending_batch, embeddings)
        else:
            vector_store.add_documents(pending_batch)
        pending_batch = []

    for path in file_paths:
        base = os.path.basename(path)
        try:
            docs = load_file_to_documents(path, excel_max_rows=excel_max_rows, excel_max_cols=excel_max_cols)
            print(f"  Loaded: {base} ({len(docs)} doc(s))")
        except Exception as e:
            print(f"  ERROR loading {base}: {e}")
            continue

        for doc in docs:
            # Split per-document to avoid creating a huge in-memory chunk list.
            for i, chunk_text in enumerate(splitter.split_text(doc.page_content)):
                if max_chunks > 0 and chunk_count >= max_chunks:
                    print(f"Reached MAX_CHUNKS={max_chunks}; stopping early.")
                    flush_batch()
                    break

                md = dict(doc.metadata)
                md["chunk"] = i
                pending_batch.append(
                    type(doc)(
                        page_content=chunk_text,
                        metadata=md,
                        id=f"{doc.id or 'doc'}_chunk_{i}",
                    )
                )
                chunk_count += 1

                if chunk_count % 500 == 0:
                    print(f"    Progress: {chunk_count} chunks embedded/indexed...")

                if len(pending_batch) >= faiss_add_batch:
                    flush_batch()

            if max_chunks > 0 and chunk_count >= max_chunks:
                break

        flush_batch()

    if vector_store is None:
        print("No chunks were created; aborting.")
        return 3

    print(f"Created {chunk_count} chunk(s)")

    os.makedirs(index_dir, exist_ok=True)
    vector_store.save_local(index_dir)

    write_manifest(
        index_dir,
        embedding_model=embedding_model_short,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        excel_max_rows=excel_max_rows,
        excel_max_cols=excel_max_cols,
        file_paths=file_paths,
    )

    print(f"Saved FAISS index to: {index_dir}")
    print("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
