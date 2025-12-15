# Advanced RAG Chatbot Guide

## 🎯 Quick Reference

### Installation (3 steps)
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download Ollama from https://ollama.ai and run it

# 3. Pull a model
ollama pull mistral

# 4. Start the chatbot
streamlit run app.py
```

---

## 🔍 Understanding RAG

**RAG = Retrieval-Augmented Generation**

1. **Retrieval**: Find relevant document chunks using embeddings + FAISS
2. **Augmentation**: Add retrieved context to the prompt
3. **Generation**: LLM generates answer based on context

**Why it works:**
- ✅ LLM doesn't need to know your documents beforehand
- ✅ Answers are grounded in your actual documents
- ✅ Lower hallucination rate
- ✅ Works with small, local models

---

## 📊 Model Selection Guide

### Mistral 7B ⭐ RECOMMENDED
```
Speed: ⚡⚡⚡ Very Fast (10-20 sec per query)
Quality: ⭐⭐⭐⭐ Very Good
VRAM: 4GB
Setup: ollama pull mistral
Best for: General use, most documents
```

### Neural Chat
```
Speed: ⚡⚡⚡ Very Fast (10-15 sec)
Quality: ⭐⭐⭐⭐ Good for chat
VRAM: 4GB
Setup: ollama pull neural-chat
Best for: Conversational responses
```

### Llama2
```
Speed: ⚡ Slower (30-60 sec)
Quality: ⭐⭐⭐⭐⭐ Excellent
VRAM: 7GB
Setup: ollama pull llama2
Best for: Complex questions, high-quality answers
```

---

## 🎨 Embedding Models Explained

**What are embeddings?**
Convert text into vectors (lists of numbers) where similar texts have similar vectors.

### all-MiniLM-L6-v2
- **Size**: Smallest
- **Speed**: Fastest
- **Use**: Quick demos, low-memory setups
- **Downside**: Less accurate for complex queries

### all-mpnet-base-v2 ⭐ RECOMMENDED
- **Size**: Medium
- **Speed**: Fast
- **Accuracy**: Excellent
- **Use**: Best balance, most documents
- **Upside**: Great quality with good speed

### distiluse-base-multilingual-cased-v2
- **Size**: Medium
- **Languages**: 50+ languages
- **Use**: Multilingual documents
- **Upside**: Handles multiple languages well

---

## ⚙️ Tuning Parameters

### chunk_size (500 default)
```
↑ Increase (600-800) for:
  - Long documents
  - Research papers
  - When answers need more context

↓ Decrease (300-400) for:
  - Short, focused documents
  - Quick Q&A
  - When memory is limited
```

### chunk_overlap (50 default)
```
↑ Increase (75-100) for:
  - Content that spans chunks
  - When important info gets cut off
  - Complex topics

↓ Decrease (20-30) for:
  - Simple documents
  - When speed matters
```

### k_documents (3 default)
```
↑ Increase (4-6) for:
  - Complex questions
  - Need comprehensive answers
  - When context is scattered

↓ Decrease (1-2) for:
  - Direct answer questions
  - Small documents
  - When speed matters
```

### temperature (0.3 default)
```
0.0 = Deterministic (always same answer)
0.3 = Focused (recommended for Q&A)
0.7 = Balanced
1.0 = Creative (randomness)
```

---

## 🐛 Troubleshooting

### Problem: "Connection refused"
```
Solution: 
1. Make sure Ollama is running
2. Check: http://localhost:11434 in browser
3. Restart Ollama if needed
```

### Problem: Very slow responses (>60 seconds)
```
Solutions (in order):
1. Switch to Mistral model (faster)
2. Reduce k_documents to 2-3
3. Reduce chunk_size to 400
4. Check system resources (CPU usage)
5. Add more RAM if using swap
```

### Problem: Out of memory error
```
Solutions:
1. Reduce chunk_size to 300
2. Use MiniLM embeddings instead
3. Use smaller model (Mistral instead of Llama2)
4. Close other applications
5. Restart your machine
```

### Problem: Poor answer quality
```
Solutions:
1. Increase k_documents to 4-5
2. Increase chunk_overlap to 75-100
3. Switch to better embedding model (MPNET)
4. Switch to Llama2 model (higher quality)
5. Check if documents are relevant
```

### Problem: Responses are too short
```
Solution in app.py, find:
llm = Ollama(model=model_name, ...)

Add:
llm = Ollama(
    model=model_name,
    temperature=temperature,
    num_predict=512  # Increase from 256
)
```

### Problem: Model not found
```
Solution:
1. Check installed models: ollama list
2. Pull the model: ollama pull mistral
3. Check internet connection
4. Try again after 1-2 minutes
```

---

## 🚀 Performance Optimization

### For Speed (Under 10 seconds)
```python
# Use this configuration in app.py
{
    "model": "mistral",
    "embedding": "all-MiniLM-L6-v2",
    "chunk_size": 400,
    "k_documents": 2,
    "temperature": 0.3
}
```

### For Quality (Best answers)
```python
# Use this configuration
{
    "model": "llama2",
    "embedding": "all-mpnet-base-v2",
    "chunk_size": 600,
    "k_documents": 5,
    "temperature": 0.4
}
```

### For Memory-Constrained (Low VRAM)
```python
# Use this configuration
{
    "model": "mistral",
    "embedding": "all-MiniLM-L6-v2",
    "chunk_size": 300,
    "k_documents": 2,
    "temperature": 0.2
}
```

---

## 📚 Advanced Features

### Custom Prompts
Edit in `app.py`:
```python
prompt_template = """Use the following pieces of context to answer the question.
Context:
{context}

Question: {question}

Your custom response format here:"""
```

### Batch Upload
Upload multiple PDFs at once - the app processes all of them together.

### Save Vector Store
To reuse processed documents:
```python
# In app.py, add:
vector_store.save_local("my_docs_index")

# Load later:
vector_store = FAISS.load_local("my_docs_index", embeddings)
```

### Change Retrieval Strategy
In app.py, modify:
```python
search_kwargs={'k': k_documents}
# to:
search_kwargs={'k': k_documents, 'fetch_k': k_documents*4}  # Better diversity
```

---

## 🎓 Best Practices

### Document Preparation
1. **Clean PDFs** - Remove scans/images if not needed
2. **Separate Large Documents** - Break into 10-20 page chunks
3. **Remove Duplicates** - Don't upload same doc twice
4. **Consistent Format** - Mix text and PDFs OK, but keep consistent

### Query Writing
1. **Be Specific** - "What is X?" better than "Tell me about it"
2. **Use Context** - "In the technical doc, what is..."
3. **Ask Follow-ups** - Ask "How?" or "Why?" based on answers
4. **Break It Down** - Multiple specific queries > one complex query

### Production Use
1. **Monitor Memory** - Check RAM/CPU usage
2. **Batch Queries** - Don't run 100 queries at once
3. **Cache Results** - Save FAISS index for repeated use
4. **Version Control** - Keep track of model versions used

---

## 📊 Example Use Cases

### Research Paper Analysis
```python
Settings:
- Model: Llama2 (better for academic)
- Chunk size: 800
- K documents: 5
- Temperature: 0.2

Query: "What are the main contributions of this paper?"
```

### Technical Documentation
```python
Settings:
- Model: Mistral (good for code)
- Chunk size: 500
- K documents: 3
- Temperature: 0.1

Query: "How do I implement feature X?"
```

### Legal Document Review
```python
Settings:
- Model: Llama2 (careful/precise)
- Chunk size: 400
- K documents: 4
- Temperature: 0.1

Query: "What are the key obligations?"
```

---

## 🔗 Useful Resources

- **Ollama Models**: https://ollama.ai/library
- **LangChain Docs**: https://python.langchain.com
- **Sentence Transformers**: https://www.sbert.net
- **FAISS**: https://faiss.ai

---

## 💡 Tips & Tricks

1. **First run is slow** - Models download on first use
2. **Subsequent runs are fast** - Models cached locally
3. **GPU speeds up embeddings** - If you have NVIDIA GPU
4. **Start simple** - Mistral + MiniLM for testing
5. **Experiment** - Try different models, find best for your docs
6. **Monitor quality** - Check source documents in each response

---

## 🚨 Common Mistakes to Avoid

1. ❌ Not checking if Ollama is running
2. ❌ Uploading identical documents multiple times
3. ❌ Using very large chunk sizes (>1000)
4. ❌ Expecting perfect answers (RAG is ~80-90% accurate)
5. ❌ Not checking source documents
6. ❌ Running on very old machines (<4GB RAM)
7. ❌ Forgetting to keep Ollama running in background

---

**Happy chatting! 🤖**
