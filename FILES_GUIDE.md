# 📦 Project Structure & Files Guide

## Files Included

### Core Application Files

#### 1. **app.py** - Main Streamlit Application
- The main chatbot interface
- Features:
  - Document upload (PDF & TXT)
  - Vector store creation with FAISS
  - Interactive chat interface
  - Source document display
  - Model/embedding configuration
  - Chat history

**Key sections:**
- Session state management
- Document loading and processing
- Vector store creation
- QA chain setup
- Chat interface

**To customize:**
- Change prompt template (lines ~165-175)
- Modify CSS styling (lines ~20-45)
- Adjust default settings (sidebar)

---

#### 2. **config.py** - Configuration & Tuning
Advanced configuration for fine-tuning performance.

**Includes:**
- LLM configurations (Mistral, Neural Chat, Llama2)
- Embedding model specifications
- Document processing parameters
- Retrieval settings
- System prompts for different use cases
- Performance optimization options
- Memory settings
- Logging configuration

**To use:**
Import and modify settings, then pass to app.py functions.

---

#### 3. **requirements.txt** - Python Dependencies
All required Python packages with pinned versions.

**Key packages:**
- `streamlit` - Web UI framework
- `langchain` - LLM orchestration
- `faiss-cpu` - Vector search
- `sentence-transformers` - Embeddings
- `pypdf` - PDF reading
- `ollama` - Local LLM integration

**To install:**
```bash
pip install -r requirements.txt
```

---

### Documentation Files

#### 4. **README.md** - Project Overview
Quick start guide and general information.

**Sections:**
- Quick start (5 minutes)
- Feature overview
- System requirements
- Model comparison
- Troubleshooting basics
- Privacy & security
- Example use cases

**Read this first!**

---

#### 5. **SETUP.md** - Installation Guide
Detailed step-by-step setup instructions.

**Sections:**
- Python dependency installation
- Ollama installation
- Model pulling
- Configuration options
- System requirements
- Troubleshooting
- Tips and tricks

**Use this for:** Installation issues, setup help

---

#### 6. **ADVANCED.md** - Advanced Features
In-depth guide for optimization and customization.

**Sections:**
- Understanding RAG
- Model selection detailed guide
- Embedding models explained
- Parameter tuning guide
- Performance optimization
- Troubleshooting advanced issues
- Best practices
- Use case examples

**Use this for:** Optimization, fine-tuning, advanced features

---

### Setup Helper Scripts

#### 7. **setup.sh** - Linux/Mac Setup Script
Automated setup for Linux and macOS users.

**Does:**
- Checks for Python
- Creates virtual environment
- Installs dependencies
- Checks for Ollama
- Provides next steps

**To run:**
```bash
chmod +x setup.sh
./setup.sh
```

---

#### 8. **setup.bat** - Windows Setup Script
Automated setup for Windows users.

**Does:**
- Checks for Python
- Creates virtual environment
- Installs dependencies
- Checks for Ollama
- Provides next steps

**To run:**
```cmd
setup.bat
```

---

## Quick File Reference

```
📁 rag-chatbot/
│
├── 🔧 Application
│   ├── app.py                  # Main chatbot (EDIT THIS for customization)
│   ├── config.py               # Advanced settings (EDIT FOR TUNING)
│   └── requirements.txt        # Dependencies (RUN: pip install -r)
│
├── 📚 Documentation
│   ├── README.md               # START HERE (quick overview)
│   ├── SETUP.md                # Installation guide
│   └── ADVANCED.md             # Advanced features & optimization
│
└── 🚀 Setup Scripts
    ├── setup.sh                # Linux/Mac setup (RUN ONCE)
    └── setup.bat               # Windows setup (RUN ONCE)
```

---

## 🔄 Typical Workflow

### First Time Setup

1. **Read**: `README.md` (2 min)
2. **Run**: `setup.sh` or `setup.bat` (5 min)
3. **Install**: Ollama from https://ollama.ai (10 min)
4. **Pull**: Model with `ollama pull mistral` (5 min)
5. **Start**: Run `streamlit run app.py` (1 min)
6. **Use**: Upload docs and ask questions!

---

### Customization

1. **Basic**: Modify UI/prompts in `app.py`
2. **Advanced**: Adjust parameters in `config.py`
3. **Help**: Check `ADVANCED.md` for guidance

---

### Troubleshooting

1. **Quick issues**: Check `README.md` troubleshooting
2. **Setup issues**: Check `SETUP.md`
3. **Performance**: Check `ADVANCED.md`
4. **Still stuck**: Review error messages, check Ollama is running

---

## 📝 File Modification Guide

### To Change the Prompt Template
**File**: `app.py` (around line 165)
```python
prompt_template = """Your custom prompt here..."""
```

### To Change Document Chunk Size
**File**: `app.py` (around line 140)
```python
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,  # Change this
    chunk_overlap=50
)
```

### To Add More Embedding Models
**File**: `app.py` (around line 70)
```python
embedding_model = st.sidebar.selectbox(
    "Select Embedding Model",
    [
        "all-MiniLM-L6-v2",
        "your-model-name"  # Add here
    ]
)
```

### To Use Advanced Configuration
**File**: `config.py`
- Modify settings in `LLM_CONFIG`, `EMBEDDING_MODELS`, etc.
- Import and use in `app.py`

---

## 🎯 Common Tasks

### Task: Make chatbot faster
1. Open `app.py`
2. Change model to `mistral` in sidebar option
3. Or modify `ADVANCED.md` performance tips

### Task: Improve answer quality
1. Read `ADVANCED.md` section "For Quality"
2. Modify chunk_size and k_documents in config
3. Use better embedding model (MPNET)

### Task: Save processed documents
1. In `app.py`, add after vector store creation:
```python
vector_store.save_local("my_docs_index")
```
2. Load later:
```python
vector_store = FAISS.load_local("my_docs_index", embeddings)
```

### Task: Change default model
1. Open `app.py`
2. Find line: `model_name = st.sidebar.selectbox(...)`
3. Change default: `value="mistral"` to another model

---

## 🚨 Important Notes

### Before Starting
- ✅ Python 3.8+ installed
- ✅ 8GB+ RAM available
- ✅ Ollama installed and running
- ✅ A model pulled (e.g., `ollama pull mistral`)

### During Use
- ✅ Keep Ollama running in background
- ✅ Check source documents for verification
- ✅ Clean PDFs work better (not scanned)

### Performance
- ⚡ First query is slow (models loading)
- ⚡ Subsequent queries are fast
- ⚡ GPU optional but helpful

---

## 📞 Frequently Viewed Sections

| Question | Where to Look |
|----------|--------------|
| "How do I install this?" | `SETUP.md` |
| "How do I use the chatbot?" | `README.md` + Quick Start |
| "Why is it slow?" | `ADVANCED.md` - Performance section |
| "How do I improve quality?" | `ADVANCED.md` - Tuning section |
| "What models can I use?" | `ADVANCED.md` - Model comparison |
| "I have an error!" | `SETUP.md` - Troubleshooting |
| "How do I customize it?" | `app.py` comments + `config.py` |

---

## 🔗 External Resources

### Official Documentation
- **Ollama**: https://ollama.ai
- **LangChain**: https://python.langchain.com
- **Streamlit**: https://docs.streamlit.io
- **FAISS**: https://faiss.ai

### Tutorials & Guides
- **Sentence Transformers**: https://www.sbert.net
- **PyPDF**: https://github.com/py-pdf/pypdf

---

## 💡 Tips for Success

1. **Start Simple**: Use default settings first, then tweak
2. **Check Sources**: Always verify answers in source docs
3. **Clean Data**: Good documents = good answers
4. **Experiment**: Try different models/settings
5. **Read Docs**: Most issues are covered in guides
6. **Monitor Resources**: Check RAM/CPU while running

---

**You now have a complete, production-ready RAG chatbot!** 🚀

Next step: Follow the Quick Start in `README.md`
