# 🎯 RAG Chatbot - Quick Reference Card

## 🚀 30-Second Start
```bash
pip install -r requirements.txt
# Install Ollama from https://ollama.ai
ollama pull mistral
streamlit run app.py
# Open http://localhost:8501
```

---

## 📋 Essential Commands

### Installation
```bash
pip install -r requirements.txt          # Install Python packages
ollama pull mistral                       # Download model
```

### Running
```bash
streamlit run app.py                     # Start chatbot (opens auto)
```

### Checking Status
```bash
ollama list                               # See installed models
python -c "import streamlit; print('OK')" # Check Streamlit
```

---

## ⚙️ Key Settings (in UI Sidebar)

| Setting | Options | Default | Effect |
|---------|---------|---------|--------|
| **Model** | mistral, neural-chat, llama2 | mistral | Speed vs quality |
| **Temperature** | 0.0-1.0 | 0.3 | Focused vs creative |
| **Embedding** | MiniLM, MPNET, Multilingual | MiniLM | Retrieval quality |
| **Documents** | 1-10 | 3 | Context size |

---

## 📁 Files at a Glance

| File | Purpose | Edit for... |
|------|---------|------------|
| `app.py` | Main application | Prompts, UI, features |
| `config.py` | Advanced settings | Fine-tuning, optimization |
| `requirements.txt` | Dependencies | Adding packages |
| `README.md` | Overview | Read first |
| `SETUP.md` | Installation | Setup help |
| `ADVANCED.md` | Advanced guide | Optimization tips |

---

## 🎓 Model Quick Comparison

```
Mistral 7B ⭐
├─ Speed: ⚡⚡⚡ Very fast
├─ Quality: ⭐⭐⭐⭐ Very good
├─ VRAM: 4GB
└─ Best for: General use

Neural Chat
├─ Speed: ⚡⚡⚡ Very fast  
├─ Quality: ⭐⭐⭐⭐ Good
├─ VRAM: 4GB
└─ Best for: Chat responses

Llama2
├─ Speed: ⚡ Slow
├─ Quality: ⭐⭐⭐⭐⭐ Best
├─ VRAM: 7GB
└─ Best for: Complex questions
```

---

## 🔌 Port Reference

| Service | Port | Status Check |
|---------|------|-------------|
| Ollama | 11434 | http://localhost:11434 |
| Streamlit | 8501 | http://localhost:8501 |

---

## 🆘 Quick Troubleshooting

| Issue | Solution |
|-------|----------|
| "Connection refused" | Start Ollama application |
| "Model not found" | `ollama pull mistral` |
| "Very slow" | Use Mistral model, reduce docs to 2 |
| "Out of memory" | Use MiniLM embeddings, reduce chunk_size |
| "Poor answers" | Increase docs to 4-5, use MPNET embeddings |

---

## 🎯 Optimization Profiles

### ⚡ Speed (< 10 sec per query)
```
Model: mistral
Embedding: all-MiniLM-L6-v2
Chunk size: 400
Documents: 2
```

### ⭐ Quality (best answers)
```
Model: llama2
Embedding: all-mpnet-base-v2
Chunk size: 600
Documents: 5
```

### 💡 Balanced (recommended)
```
Model: mistral
Embedding: all-mpnet-base-v2
Chunk size: 500
Documents: 3
```

---

## 📊 How RAG Works (Simple)

```
User: "What does document say about X?"
   ↓
Vector Search: Find chunks similar to query
   ↓
Retrieved: Top 3 relevant chunks
   ↓
Prompt: "Answer based on: [chunks]"
   ↓
LLM: Generates answer from context
   ↓
Result: Answer + source documents
```

---

## 💾 File Operations

### Save Vector Index (for reuse)
```python
# In app.py after vector_store creation:
vector_store.save_local("my_index")
```

### Load Saved Index
```python
from langchain.vectorstores import FAISS
vector_store = FAISS.load_local("my_index", embeddings)
```

---

## 🎨 Customization Quick Tips

### Change Prompt
File: `app.py` line ~165
```python
prompt_template = """Your custom prompt..."""
```

### Change Chunk Size
File: `app.py` line ~140
```python
chunk_size=500  # Increase for longer context
```

### Change Default Model
File: `app.py` line ~70
```python
st.sidebar.selectbox(..., value="mistral")  # Change to llama2
```

---

## 📈 Performance Metrics

| Configuration | Speed | Quality | Memory |
|---------------|-------|---------|--------|
| Mistral + MiniLM | ⚡⚡⚡ | ⭐⭐⭐ | 💾 Low |
| Mistral + MPNET | ⚡⚡ | ⭐⭐⭐⭐ | 💾 Medium |
| Llama2 + MPNET | ⚡ | ⭐⭐⭐⭐⭐ | 💾 High |

---

## 🔐 Privacy Checklist

- ✅ All processing local (no cloud)
- ✅ No internet after setup (except model downloads)
- ✅ Documents stay on your machine
- ✅ No tracking or telemetry
- ✅ Open source (inspect code)

---

## 📱 Example Queries

### Research Document
> "What are the 3 main findings?"
> "What methodology was used?"

### Technical Doc
> "How do I set up X?"
> "What are system requirements?"

### Book/Article
> "Summarize chapter 2"
> "What happens when..."

### Legal Document
> "What are my obligations?"
> "Are there penalties for..."

---

## ⏱️ Typical Times

| Operation | Duration | Hardware |
|-----------|----------|----------|
| First query | 30-60 sec | Mistral, first load |
| Subsequent | 5-15 sec | Mistral, warm |
| With Llama2 | 30-90 sec | More powerful |
| Embedding docs | 5-30 sec | Depends on size |

---

## 🚀 Next Steps After Installation

1. ✅ Test with sample PDF (verify it works)
2. ✅ Try different models (find your preference)
3. ✅ Read `ADVANCED.md` (optimize for your use case)
4. ✅ Customize prompt (tailor to your domain)
5. ✅ Save vector index (reuse processed docs)

---

## 📞 Where to Find Help

| Question | File |
|----------|------|
| "How to install?" | `SETUP.md` |
| "How to use?" | `README.md` |
| "Why is it slow?" | `ADVANCED.md` |
| "How to customize?" | `app.py` + `config.py` |
| "Detailed guide?" | `FILES_GUIDE.md` |

---

## 🎉 You're Ready!

1. Run `setup.sh` or `setup.bat`
2. Keep Ollama running
3. Run `streamlit run app.py`
4. Upload documents
5. Start chatting!

**Happy RAG-ing! 🤖**

---

## System Requirements TL;DR

- **Python**: 3.8+
- **RAM**: 8GB minimum
- **Disk**: 5GB for Mistral
- **Internet**: Download only

---

**Last Updated**: December 2024
**Version**: 1.0.0
**Status**: Production Ready ✅
