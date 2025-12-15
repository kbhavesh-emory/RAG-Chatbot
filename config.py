"""
Advanced Configuration for RAG Chatbot
Modify these settings to optimize performance for your use case
"""

# LLM Configuration
LLM_CONFIG = {
    "mistral": {
        "model_name": "mistral",
        "base_url": "http://localhost:11434",
        "temperature": 0.3,
        "top_p": 0.9,
        "num_predict": 256,  # Max tokens to generate
        "context_window": 8192
    },
    "neural-chat": {
        "model_name": "neural-chat",
        "base_url": "http://localhost:11434",
        "temperature": 0.3,
        "top_p": 0.95,
        "num_predict": 256,
        "context_window": 4096
    },
    "llama2": {
        "model_name": "llama2",
        "base_url": "http://localhost:11434",
        "temperature": 0.5,
        "top_p": 0.9,
        "num_predict": 512,
        "context_window": 4096
    }
}

# Embedding Models - Ranked by quality vs speed
EMBEDDING_MODELS = {
    "all-MiniLM-L6-v2": {
        "name": "sentence-transformers/all-MiniLM-L6-v2",
        "dimensions": 384,
        "speed": "⚡⚡⚡ Very Fast",
        "quality": "Good",
        "vram": "~1GB",
        "best_for": "Quick retrieval, low-memory setups"
    },
    "all-mpnet-base-v2": {
        "name": "sentence-transformers/all-mpnet-base-v2",
        "dimensions": 768,
        "speed": "⚡⚡ Fast",
        "quality": "Very Good",
        "vram": "~2GB",
        "best_for": "Balanced quality and speed (recommended)"
    },
    "distiluse-base-multilingual-cased-v2": {
        "name": "sentence-transformers/distiluse-base-multilingual-cased-v2",
        "dimensions": 512,
        "speed": "⚡ Moderate",
        "quality": "Excellent",
        "vram": "~2GB",
        "best_for": "Multilingual documents"
    }
}

# Document Processing
DOCUMENT_PROCESSING = {
    "chunk_size": 500,           # Characters per chunk (adjust for domain)
    "chunk_overlap": 50,         # Overlap between chunks (prevents cutting important info)
    "separators": ["\n\n", "\n", " ", ""],  # Split priority
    "strip_whitespace": True,
    "remove_empty_lines": True
}

# Retrieval Settings
RETRIEVAL_CONFIG = {
    "k_documents": 3,            # Number of chunks to retrieve
    "score_threshold": 0.5,      # Minimum similarity score (0-1)
    "return_source_docs": True,  # Show source documents
    "search_type": "similarity"  # or "mmr" for diversity
}

# Prompts for different use cases
SYSTEM_PROMPTS = {
    "default": """You are a helpful assistant answering questions based on provided documents.
- Answer only from the provided context
- If you don't know, say "I don't have enough information"
- Be concise and clear
- Cite relevant parts from the documents""",
    
    "research": """You are a research assistant. Provide detailed, thorough answers.
- Include specific details and examples from documents
- If relevant, explain context and background
- Highlight important connections between concepts
- Acknowledge limitations in the provided documents""",
    
    "summarization": """You are a summarization expert.
- Create clear, concise summaries
- Highlight key points and main ideas
- Maintain important details but remove redundancy
- Organize information logically""",
    
    "technical": """You are a technical assistant.
- Explain technical concepts clearly
- Include relevant code examples or specifications
- Use precise terminology
- Break down complex topics into understandable parts"""
}

# Output formatting
OUTPUT_FORMAT = {
    "max_length": 500,           # Maximum response length
    "include_sources": True,     # Show source documents
    "format_citations": True,    # Format citations in response
    "show_confidence": False,    # Show similarity scores
}

# Performance Tuning
PERFORMANCE = {
    "cache_embeddings": True,    # Cache embeddings for faster reload
    "batch_processing": True,    # Process multiple chunks at once
    "parallel_jobs": 4,          # Number of parallel jobs for embeddings
    "device": "cpu"              # "cpu" or "cuda" (if GPU available)
}

# Advanced Settings
ADVANCED = {
    "use_mmr": False,            # Maximum Marginal Relevance (reduces duplication)
    "diversity_penalty": 0.5,    # For MMR search (0-1)
    "rerank_results": False,     # Rerank results by relevance
    "use_hyde": False,           # Hypothetical Document Embeddings
    "enable_summary": True,      # Generate summaries of chunks
    "min_chunk_length": 100      # Minimum chunk length to avoid fragments
}

# Sample Processing Parameters by Document Type
DOCUMENT_TYPE_SETTINGS = {
    "research_paper": {
        "chunk_size": 800,
        "chunk_overlap": 100,
        "k_documents": 5,
        "prompt": SYSTEM_PROMPTS["research"]
    },
    "technical_documentation": {
        "chunk_size": 500,
        "chunk_overlap": 50,
        "k_documents": 3,
        "prompt": SYSTEM_PROMPTS["technical"]
    },
    "book": {
        "chunk_size": 600,
        "chunk_overlap": 75,
        "k_documents": 4,
        "prompt": SYSTEM_PROMPTS["default"]
    },
    "legal_document": {
        "chunk_size": 400,
        "chunk_overlap": 100,
        "k_documents": 5,
        "prompt": SYSTEM_PROMPTS["default"]
    }
}

# Memory optimization for large documents
MEMORY_OPTIMIZATION = {
    "enable_streaming": True,    # Stream responses instead of waiting
    "max_cached_documents": 10,  # Maximum documents to keep in memory
    "cleanup_interval": 3600,    # Clean cache every hour
    "compress_embeddings": False # Compress large embeddings
}

# Logging and debugging
LOGGING = {
    "enable_logging": False,
    "log_file": "rag_chatbot.log",
    "log_queries": True,         # Log user queries
    "log_sources": True,         # Log retrieved sources
    "log_timing": True           # Log response times
}

# Default UI Settings
UI_SETTINGS = {
    "theme": "light",            # "light" or "dark"
    "max_chat_history": 50,      # Keep last 50 messages
    "show_source_preview": True, # Show source document previews
    "preview_length": 500,       # Characters to preview
    "auto_scroll": True,         # Auto-scroll to latest message
}

# Optimization Tips:
# 1. For faster responses: Use MiniLM embeddings + Mistral 7B
# 2. For better quality: Use MPNET embeddings + Llama2
# 3. For multilingual: Use distiluse-multilingual
# 4. For large documents: Increase chunk_overlap, reduce k_documents
# 5. For specific domains: Create custom prompts and tune chunk_size
