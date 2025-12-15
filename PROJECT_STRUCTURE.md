# 📁 RAG Chatbot - Complete Project File Structure

## Directory Layout

```
rag-chatbot/
│
├── 📂 Core Application
│   ├── app.py                    # Main Streamlit chatbot application
│   ├── config.py                 # Advanced configuration settings
│   └── requirements.txt          # Python dependencies
│
├── 📂 Documentation
│   ├── README.md                 # Quick start & project overview ⭐ START HERE
│   ├── SETUP.md                  # Step-by-step installation guide
│   ├── ADVANCED.md               # Advanced features & optimization
│   ├── FILES_GUIDE.md            # Detailed file reference
│   ├── QUICK_REFERENCE.md        # Cheat sheet & quick commands
│   └── PROJECT_STRUCTURE.md      # This file
│
├── 📂 Setup Scripts
│   ├── setup.sh                  # Linux/Mac automated setup
│   └── setup.bat                 # Windows automated setup
│
├── 📂 Runtime Directories (Created when running)
│   ├── venv/                     # Virtual environment (after setup)
│   ├── temp_*/                   # Temporary files during processing
│   └── faiss_index/              # Saved vector indexes (optional)
│
└── 📂 Optional - User Data
    ├── documents/                # Your uploaded PDFs/TXT files
    └── saved_indexes/            # Saved FAISS indexes
```

---

## 📋 File Descriptions

### Core Application Files

#### `app.py` (Main Application) - ~450 lines
**Purpose**: Complete Streamlit chatbot interface

**Key Sections**:
- Imports & configuration (lines 1-20)
- Streamlit page setup (lines 22-30)
- Custom CSS styling (lines 32-60)
- Session state initialization (lines 62-75)
- Sidebar configuration (lines 77-120)
- Helper functions (lines 122-200)
- Document processing (lines 202-250)
- Vector store creation (lines 252-280)
- QA chain setup (lines 282-310)
- Main UI layout (lines 312-450)

**Can Edit For**:
- Prompt template customization
- UI styling and layout
- Default settings
- Features and functionality

---

#### `config.py` (Configuration) - ~200 lines
**Purpose**: Advanced tuning and optimization settings

**Sections**:
- LLM configurations (lines 10-30)
- Embedding models (lines 32-55)
- Document processing settings (lines 57-65)
- Retrieval configuration (lines 67-73)
- System prompts by use case (lines 75-110)
- Output formatting (lines 112-118)
- Performance tuning (lines 120-127)
- Advanced settings (lines 129-137)
- Document type presets (lines 139-165)
- Memory optimization (lines 167-175)
- Logging settings (lines 177-185)
- UI settings (lines 187-195)

**Can Edit For**:
- Fine-tuning performance
- Custom prompts for different domains
- Model-specific settings
- Optimization parameters

---

#### `requirements.txt` (Dependencies) - 8 lines
**Purpose**: Python package specifications

**Contains**:
```
streamlit==1.32.0           # Web UI framework
langchain==0.1.9            # LLM orchestration
langchain-community==0.0.19 # Community integrations
faiss-cpu==1.7.4            # Vector search database
sentence-transformers==2.2.2 # Embedding models
pypdf==3.17.1               # PDF reading
python-dotenv==1.0.0        # Environment variables
ollama==0.1.12              # Ollama integration
```

**Can Edit For**:
- Adding new Python packages
- Version updates
- Additional libraries

---

### Documentation Files

#### `README.md` (Project Overview) - ~350 lines
**Purpose**: Quick start guide and feature overview

**Sections**:
- Project description
- Key features list
- Quick start (5 steps)
- Project structure
- How it works
- System requirements
- Model comparison
- Troubleshooting
- Privacy & security
- Use cases
- Resources & links

**Read When**: First time setup, understanding project

---

#### `SETUP.md` (Installation Guide) - ~300 lines
**Purpose**: Detailed step-by-step installation

**Sections**:
- Quick start steps
- Feature overview
- Configuration options
- System requirements
- RAG explanation
- Model comparison
- Embedding models
- Troubleshooting
- Tips & tricks
- Customization guide

**Read When**: Installation help, setup issues

---

#### `ADVANCED.md` (Advanced Features) - ~400 lines
**Purpose**: Advanced customization and optimization

**Sections**:
- Quick reference
- RAG deep dive
- Model selection guide
- Embedding models explained
- Parameter tuning
- Performance optimization
- Troubleshooting advanced issues
- Best practices
- Production use
- Use case examples
- Example configurations

**Read When**: Optimizing performance, advanced features

---

#### `FILES_GUIDE.md` (File Reference) - ~300 lines
**Purpose**: Detailed description of every file

**Sections**:
- Files included
- Core application files
- Documentation files
- Setup helper scripts
- Quick reference table
- Typical workflow
- File modification guide
- Common tasks
- Important notes
- Frequently viewed sections

**Read When**: Understanding file purposes, finding what to edit

---

#### `QUICK_REFERENCE.md` (Cheat Sheet) - ~250 lines
**Purpose**: Quick lookup reference

**Sections**:
- 30-second start
- Essential commands
- Key settings table
- Files at a glance
- Model comparison
- Port reference
- Quick troubleshooting
- Optimization profiles
- RAG explanation
- File operations
- Customization tips
- Performance metrics
- Example queries
- Next steps

**Read When**: Quick lookup, commands, troubleshooting

---

### Setup Scripts

#### `setup.sh` (Linux/Mac Setup) - ~60 lines
**Purpose**: Automated setup for Linux/Mac

**Does**:
- Checks Python installation
- Creates virtual environment
- Activates venv
- Installs dependencies
- Checks for Ollama
- Provides next steps

**Run**: `chmod +x setup.sh && ./setup.sh`

---

#### `setup.bat` (Windows Setup) - ~70 lines
**Purpose**: Automated setup for Windows

**Does**:
- Checks Python installation
- Creates virtual environment
- Activates venv
- Installs dependencies
- Checks for Ollama
- Provides next steps

**Run**: `setup.bat`

---

## 🔄 File Dependencies

```
app.py
├─ imports from: requirements.txt (dependencies)
├─ uses: config.py (optional, for settings)
├─ reads: Uploaded PDF/TXT files
├─ creates: FAISS vector store (in memory)
├─ creates: temp_* files (during processing)
└─ connects to: Ollama (localhost:11434)

config.py
├─ imported by: app.py (optional)
└─ independent file (no dependencies)

requirements.txt
├─ used by: setup.sh and setup.bat
└─ used by: pip install command

setup.sh / setup.bat
├─ installs: All packages from requirements.txt
├─ creates: venv/ directory
└─ checks: Python, Ollama installation
```

---

## 📊 File Statistics

| File | Type | Size | Lines | Purpose |
|------|------|------|-------|---------|
| app.py | Python | ~25KB | 450 | Main application |
| config.py | Python | ~12KB | 200 | Configuration |
| requirements.txt | Text | ~200B | 8 | Dependencies |
| README.md | Markdown | ~18KB | 350 | Overview |
| SETUP.md | Markdown | ~20KB | 300 | Installation |
| ADVANCED.md | Markdown | ~25KB | 400 | Advanced guide |
| FILES_GUIDE.md | Markdown | ~18KB | 300 | File reference |
| QUICK_REFERENCE.md | Markdown | ~15KB | 250 | Cheat sheet |
| setup.sh | Bash | ~2KB | 60 | Linux/Mac setup |
| setup.bat | Batch | ~2.5KB | 70 | Windows setup |

**Total**: ~140KB documentation, ~37KB code

---

## 🗂️ Runtime Directory Structure

### After First Run

```
rag-chatbot/
├── venv/                    # Virtual environment (if using venv)
│   ├── bin/ or Scripts/     # Executables
│   ├── lib/                 # Installed packages
│   └── pyvenv.cfg          # Configuration
│
├── temp_file1.pdf          # Temporary files during upload
├── temp_file2.txt
│
├── .streamlit/             # Streamlit config (auto-created)
│   └── config.toml
│
└── faiss_index/            # Optional saved indexes
    ├── index.faiss
    └── index.pkl
```

---

## 📝 Editing Guide

### What to Edit - Quick Reference

| Goal | File | Line# | What |
|------|------|-------|------|
| Change prompt | app.py | ~165 | `prompt_template` |
| Change chunk size | app.py | ~140 | `chunk_size=500` |
| Change default model | app.py | ~70 | `value="mistral"` |
| Change CSS styling | app.py | ~32-60 | `<style>` section |
| Add embedding model | app.py | ~95 | selectbox options |
| Adjust temperature | app.py | ~75-80 | slider defaults |
| Fine-tune settings | config.py | Any | Modify dictionaries |
| Add dependency | requirements.txt | End | Add new line |

---

## 🔒 File Permissions

### Important Notes

```
✅ Can Modify:
  - app.py (customize features)
  - config.py (adjust settings)
  - requirements.txt (add packages)
  - README.md (update docs)
  - Any .md files

⚠️  Be Careful:
  - setup.sh / setup.bat (affects installation)
  - Don't break imports in app.py

🚫 Don't Delete:
  - requirements.txt (dependencies list)
  - Any core files (breaks app)
```

---

## 📦 Backup Checklist

**Files to Backup Before Major Changes**:
- ✅ app.py (if making major edits)
- ✅ config.py (if customizing extensively)
- ✅ requirements.txt (if adding packages)

**Files Safe to Delete**:
- ✅ temp_* files (recreated automatically)
- ✅ venv/ directory (can be recreated)
- ✅ .streamlit/ directory (auto-recreated)

---

## 🚀 Quick File Lookup

| You need to... | Go to file | Section |
|---|---|---|
| Understand project | README.md | Overview |
| Install the project | SETUP.md | Quick Start |
| Use the chatbot | README.md | Quick Start |
| Optimize performance | ADVANCED.md | Tuning |
| Find a command | QUICK_REFERENCE.md | Commands |
| Modify prompt | app.py | Line ~165 |
| Add Python package | requirements.txt | End of file |
| Understand files | FILES_GUIDE.md | File descriptions |
| Change default model | app.py | Line ~70 |
| Custom configuration | config.py | Any section |

---

## 📈 Typical File Growth

As you use the app:

```
Initial:           ~140KB (docs) + ~37KB (code)
After 1st run:    +~50MB (Ollama models cache)
After documents:  +~1-100MB (FAISS indexes, depends on doc size)
Chat history:     +negligible (stored in memory only)
```

---

## 🎯 Key Paths

### Python Imports
```python
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import Ollama
import streamlit as st
```

### Model Downloads
```
Ollama models: ~/.ollama/models/  (auto-managed)
Embeddings: ~/.cache/huggingface/  (auto-managed)
```

### Web Access
```
Streamlit UI: http://localhost:8501
Ollama API: http://localhost:11434
```

---

## 📋 File Checklist

Before running, ensure you have:

- [ ] `app.py` - Main application
- [ ] `config.py` - Configuration
- [ ] `requirements.txt` - Dependencies
- [ ] At least one `.md` file for reference
- [ ] Python 3.8+ installed
- [ ] Ollama installed and running
- [ ] A model pulled (`ollama pull mistral`)

---

## 🔗 File Relationships

```
User runs: streamlit run app.py
           ↓
app.py loads requirements (via imports)
           ↓
app.py reads config.py (optional)
           ↓
app.py connects to Ollama (localhost:11434)
           ↓
User uploads documents
           ↓
app.py uses PyPDF/TextLoader
           ↓
app.py creates embeddings via HuggingFace
           ↓
app.py stores in FAISS
           ↓
User asks question
           ↓
app.py retrieves from FAISS
           ↓
app.py sends to Ollama
           ↓
Display result + sources
```

---

**Version**: 1.0.0  
**Last Updated**: December 2024  
**Status**: Production Ready ✅
