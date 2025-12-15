"""
Lightweight RAG Chatbot - FIXED WITH ID FIELD
Added id field to SimpleDocument for FAISS compatibility
"""

import streamlit as st
import os
from typing import List
from dataclasses import dataclass, field
import uuid


# Import ONLY from langchain_community what actually exists
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama


# ==================== CUSTOM DOCUMENT CLASS ====================
@dataclass
class SimpleDocument:
    """Simple document class with all required fields"""
    page_content: str
    metadata: dict
    id: str = field(default_factory=lambda: str(uuid.uuid4()))


# ==================== CUSTOM TEXT SPLITTER ====================
class SimpleTextSplitter:
    """Simple text splitter"""
    
    def __init__(self, chunk_size=500, chunk_overlap=50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    
    def split_text(self, text: str) -> List[str]:
        """Split text into chunks"""
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self.chunk_size
            
            if end < len(text):
                for delimiter in ['. ', '\n\n', '\n', ' ']:
                    last_pos = text.rfind(delimiter, start, end)
                    if last_pos > start:
                        end = last_pos + len(delimiter)
                        break
            
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            
            start = end - self.chunk_overlap
        
        return chunks
    
    def split_documents(self, documents) -> List[SimpleDocument]:
        """Split documents into chunks"""
        split_docs = []
        
        for doc in documents:
            text_chunks = self.split_text(doc.page_content)
            
            for i, chunk in enumerate(text_chunks):
                metadata = doc.metadata.copy()
                metadata['chunk'] = i
                split_docs.append(SimpleDocument(
                    page_content=chunk, 
                    metadata=metadata,
                    id=f"{doc.id or 'doc'}_chunk_{i}"
                ))
        
        return split_docs


# ==================== SIMPLE QA CLASS ====================
class SimpleQA:
    """Simple QA without any chain imports"""
    
    def __init__(self, vector_store, llm, k=3):
        self.vector_store = vector_store
        self.llm = llm
        self.k = k
    
    def __call__(self, query):
        """Search and answer"""
        # Search for relevant documents
        docs = self.vector_store.similarity_search(query, k=self.k)
        
        # Create context from documents
        context = "\n\n".join([doc.page_content for doc in docs])
        
        # Create prompt
        prompt = f"""Use the following pieces of context to answer the question.
If you don't know the answer, say "I don't have enough information to answer this question."

Context:
{context}

Question: {query}

Helpful Answer:"""
        
        # Get answer from LLM using .invoke() method
        try:
            # Try .invoke() method (newer LangChain)
            answer = self.llm.invoke(prompt)
        except AttributeError:
            # Fallback to .generate() method
            try:
                result = self.llm.generate([prompt])
                answer = result.generations[0][0].text
            except:
                # Last resort: direct __call__
                answer = self.llm(prompt)
        
        return {
            "result": answer,
            "source_documents": docs
        }


# ==================== STREAMLIT CONFIG ====================
st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .chat-message {
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        display: flex;
        gap: 1rem;
    }
    .chat-message.user {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
    }
    .chat-message.assistant {
        background-color: #f5f5f5;
        border-left: 4px solid #4caf50;
    }
    </style>
""", unsafe_allow_html=True)

# ==================== SESSION STATE ====================
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None
if "documents_loaded" not in st.session_state:
    st.session_state.documents_loaded = False


# ==================== SIDEBAR ====================
st.sidebar.title("⚙️ Configuration")

st.sidebar.markdown("### Document Upload")
uploaded_files = st.sidebar.file_uploader(
    "Upload PDF or Text files",
    type=["pdf", "txt"],
    accept_multiple_files=True
)

st.sidebar.markdown("### LLM Settings")
model_name = st.sidebar.selectbox(
    "Select Ollama Model",
    ["mistral", "neural-chat", "llama2"]
)

temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.3)

st.sidebar.markdown("### Embedding Model")
embedding_model = st.sidebar.selectbox(
    "Select Embedding Model",
    ["all-MiniLM-L6-v2", "all-mpnet-base-v2"]
)

k_documents = st.sidebar.slider("Documents to retrieve", 1, 10, 3)


# ==================== HELPER FUNCTIONS ====================
@st.cache_resource
def load_embeddings(model_name: str):
    """Load embedding model"""
    return HuggingFaceEmbeddings(model_name=f"sentence-transformers/{model_name}")


@st.cache_resource
def load_llm(model_name: str, temperature: float):
    """Load Ollama LLM"""
    return Ollama(
        model=model_name,
        temperature=temperature,
        base_url="http://localhost:11434"
    )


def load_documents(uploaded_files):
    """Load documents from uploaded files"""
    documents = []
    
    for uploaded_file in uploaded_files:
        temp_path = f"temp_{uploaded_file.name}"
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        try:
            if uploaded_file.type == "application/pdf":
                loader = PyPDFLoader(temp_path)
            else:
                loader = TextLoader(temp_path)
            
            loaded_docs = loader.load()
            
            # Convert to SimpleDocument with IDs
            for doc in loaded_docs:
                documents.append(SimpleDocument(
                    page_content=doc.page_content,
                    metadata=doc.metadata,
                    id=str(uuid.uuid4())
                ))
            
            st.success(f"✅ Loaded: {uploaded_file.name}")
        except Exception as e:
            st.error(f"❌ Error loading {uploaded_file.name}: {str(e)}")
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)
    
    return documents


def create_vector_store(documents, embeddings):
    """Create FAISS vector store using custom splitter"""
    splitter = SimpleTextSplitter(chunk_size=500, chunk_overlap=50)
    text_chunks = splitter.split_documents(documents)
    
    st.info(f"📊 Created {len(text_chunks)} text chunks")
    
    vector_store = FAISS.from_documents(text_chunks, embeddings)
    return vector_store


# ==================== MAIN APP ====================
st.title("🤖 RAG Chatbot")
st.markdown("*Ask questions about your documents - powered by Ollama + FAISS*")

# Check Ollama
ollama_running = False
try:
    test_llm = Ollama(model=model_name, base_url="http://localhost:11434")
    # Try to invoke it properly
    try:
        test_llm.invoke("test")
        ollama_running = True
    except:
        try:
            test_llm("test")
            ollama_running = True
        except:
            ollama_running = False
except Exception as e:
    ollama_running = False

if not ollama_running:
    st.error(f"""
    ⚠️ **Ollama is not running!**
    
    Please install and start Ollama:
    1. Download from: https://ollama.ai
    2. Install and run it
    3. Pull a model: `ollama pull {model_name}`
    4. Keep Ollama running in the background
    5. Refresh this page once Ollama is running
    """)

# Document upload
col1, col2 = st.columns([3, 1])

with col1:
    st.markdown("### 📄 Upload Documents")
    if uploaded_files:
        st.markdown(f"**Files selected:** {len(uploaded_files)}")

with col2:
    process_button = st.button("🔄 Process Files", use_container_width=True)

if process_button and uploaded_files:
    with st.spinner("📥 Loading documents..."):
        documents = load_documents(uploaded_files)
    
    if documents:
        with st.spinner("🔗 Creating embeddings..."):
            embeddings = load_embeddings(embedding_model)
            st.session_state.vector_store = create_vector_store(documents, embeddings)
        
        with st.spinner("🔨 Building QA chain..."):
            llm = load_llm(model_name, temperature)
            st.session_state.qa_chain = SimpleQA(
                st.session_state.vector_store, llm, k=k_documents
            )
        
        st.session_state.documents_loaded = True
        st.success("✅ Documents processed! Ready to answer questions.")
        st.balloons()

elif process_button and not uploaded_files:
    st.warning("⚠️ Please upload at least one file first!")

# Chat interface
st.markdown("---")
st.markdown("### 💬 Chat")

if not st.session_state.documents_loaded:
    st.info("📌 Please upload and process documents to start chatting!")
else:
    # Display chat history
    for message in st.session_state.messages:
        role = message["role"]
        content = message["content"]
        
        if role == "user":
            st.markdown(f"""
            <div class="chat-message user">
                <strong>You:</strong><br>{content}
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="chat-message assistant">
                <strong>Assistant:</strong><br>{content}
            </div>
            """, unsafe_allow_html=True)
    
    user_input = st.chat_input("Ask a question about your documents...")
    
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        
        with st.spinner("🔍 Searching documents..."):
            try:
                response = st.session_state.qa_chain(user_input)
                answer = response["result"]
                
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer
                })
                
                if "source_documents" in response:
                    with st.expander("📚 Source Documents"):
                        for i, doc in enumerate(response["source_documents"], 1):
                            st.markdown(f"**Source {i}:**")
                            st.text_area(
                                f"content_{i}",
                                value=doc.page_content[:500] + "...",
                                height=100,
                                disabled=True,
                                label_visibility="collapsed"
                            )
                
                st.rerun()
            except Exception as e:
                st.error(f"Error: {str(e)}")

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; font-size: 0.9rem;">
    <p>🚀 Lightweight RAG Chatbot | 100% Local | No API Keys</p>
</div>
""", unsafe_allow_html=True)
