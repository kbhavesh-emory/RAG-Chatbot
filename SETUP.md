# RAG Chatbot - Setup Guide

A lightweight, free, locally-running RAG (Retrieval-Augmented Generation) chatbot using Ollama, FAISS, and Sentence Transformers.

## 🚀 Quick Start

### Step 1: Install Python Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Install and Run Ollama

1. **Download Ollama** from https://ollama.ai
2. **Install** and launch the application
3. **Pull a model** (run in terminal/command prompt):

```bash
# Mistral 7B (recommended - fastest)
ollama pull mistral

# Or alternatives:
ollama pull neural-chat   # Optimized for chat
ollama pull llama2         # More powerful but slower
```

Keep Ollama running in the background while using the chatbot.

### Step 3: Run the Chatbot

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

---

## 📊 Features

✅ **Local Processing** - Everything runs on your machine, no cloud APIs needed  
✅ **PDF & Text Support** - Upload PDFs and text files  
✅ **Smart Retrieval** - Finds most relevant document chunks  
✅ **Multiple Models** - Choose between Mistral, Neural Chat, or Llama2  
✅ **Customizable** - Adjust temperature and retrieval parameters  
✅ **Source Attribution** - See which documents the answer came from  
✅ **Chat History** - Maintains conversation context  

---

## 🔧 Configuration Options

### In the Sidebar:

- **Model Selection**: Choose your LLM (Mistral recommended for speed)
- **Temperature**: 0 = focused answers, 1 = creative answers
- **Embedding Model**: Choose lightweight sentence embeddings
- **Documents to Retrieve**: How many relevant chunks to use (3-5 recommended)

---

## 💻 System Requirements

- **RAM**: 8GB minimum (16GB recommended)
- **Disk**: 5GB for Mistral 7B model
- **GPU**: Optional (CPU works fine for Mistral)
- **Internet**: Only needed for first-time model download

---

## 🎯 How It Works

1. **Upload Documents** → PDFs/TXT files
2. **Process** → Split into chunks, create embeddings using Sentence Transformers
3. **Store** → FAISS vector database indexes embeddings
4. **Query** → Find similar chunks from your documents
5. **Generate** → Ollama LLM generates answer based on retrieved chunks

---

## 📝 Example Usage

```
User: "What are the main points from the document?"
Assistant: [Searches your documents] 
           [Generates summary based on retrieved chunks]
           [Shows source documents used]
```

---

## 🚨 Troubleshooting

### "Ollama is not running"
- Make sure Ollama app is open and running
- Check that `http://localhost:11434` is accessible

### "Model not found"
- Pull the model: `ollama pull mistral`
- Verify model is installed: `ollama list`

### "Out of memory"
- Switch to lighter model: `neural-chat` or use Mistral
- Reduce `chunk_size` in app.py (line ~140)

### "Very slow responses"
- Use CPU-optimized model like Mistral 7B
- Enable GPU if available
- Reduce temperature for faster inference

---

## 📦 What's Included

- **app.py** - Main Streamlit application
- **requirements.txt** - Python dependencies
- **setup.md** - This guide

---

## 🎨 Customization

### Change Default Chunk Size
Edit `app.py`, line ~140:
```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,  # Increase for longer chunks, decrease for shorter
    chunk_overlap=50
)
```

### Add More Embedding Models
Edit sidebar in `app.py`:
```python
embedding_model = st.sidebar.selectbox(
    "Select Embedding Model",
    [
        "all-MiniLM-L6-v2",
        "all-mpnet-base-v2",
        "your-custom-model"  # Add here
    ]
)
```

### Change Prompt Template
Edit `app.py`, lines ~165-175:
```python
prompt_template = """Your custom prompt here..."""
```

---

## 🔒 Privacy & Security

- ✅ 100% local processing - your documents never leave your machine
- ✅ No internet required after model download
- ✅ No tracking or analytics
- ✅ No API keys needed

---

## 📚 Model Comparison

| Model | Size | Speed | Quality | Best For |
|-------|------|-------|---------|----------|
| Mistral 7B | 4GB | ⚡ Fast | Good | General use, balanced |
| Neural Chat | 4GB | ⚡ Fast | Good | Chat-optimized |
| Llama2 13B | 7GB | 🐢 Slow | Better | Quality-focused |

**Recommendation**: Start with **Mistral 7B** - best balance of speed and quality.

---

## 🎓 Learn More

- LangChain: https://python.langchain.com
- Ollama: https://ollama.ai
- FAISS: https://github.com/facebookresearch/faiss
- Sentence Transformers: https://www.sbert.net

---

## 📄 License

This project uses open-source libraries under their respective licenses (MIT, Apache 2.0, etc.)

---

## 💡 Tips & Tricks

1. **Better Results**: Upload related documents together for better context
2. **Speed Up**: Use shorter documents (break large PDFs into parts)
3. **Cost Efficient**: Run on CPU - no GPU needed for Mistral
4. **Batch Processing**: Upload multiple files at once
5. **Custom Models**: Pull other Ollama models for different purposes

---

**Questions?** Check the GitHub issues or create your own!
