import streamlit as st
import os
from pathlib import Path
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import RetrievalQA

# Load environment variables
load_dotenv()

# Configuration
FAISS_INDEX_FILE = "faiss_index"
SAMPLE_DATA_DIR = "sample_data"

# Page config
st.set_page_config(
    page_title="📄 Document Analyzer Agent",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2ca02c;
        margin-top: 1.5rem;
        margin-bottom: 0.5rem;
    }
    .answer-box {
        background-color: #f0f8ff;
        border-left: 4px solid #1f77b4;
        padding: 1rem;
        border-radius: 4px;
        margin-top: 1rem;
    }
    .metadata-box {
        background-color: #fff5e6;
        border-left: 4px solid #ff7f0e;
        padding: 0.75rem;
        border-radius: 4px;
        margin-top: 0.5rem;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# Sidebar for settings
with st.sidebar:
    st.header("⚙️ Settings")
    
    # Model selection
    model_name = st.selectbox(
        "Select LLM Model",
        ["gpt-4o-mini", "gpt-4", "gpt-3.5-turbo"],
        index=0
    )
    
    # Temperature slider
    temperature = st.slider(
        "Temperature (Creativity)",
        min_value=0.0,
        max_value=1.0,
        value=0.0,
        step=0.1,
        help="Lower = more focused, Higher = more creative"
    )
    
    # Number of retrieved documents
    num_docs = st.slider(
        "Number of Retrieved Chunks",
        min_value=1,
        max_value=5,
        value=3,
        help="How many relevant chunks to retrieve"
    )
    
    st.divider()
    
    # Chunk settings
    st.subheader("🔧 Chunk Settings")
    chunk_size = st.number_input(
        "Chunk Size",
        min_value=100,
        max_value=2000,
        value=500,
        step=100
    )
    chunk_overlap = st.number_input(
        "Chunk Overlap",
        min_value=0,
        max_value=500,
        value=50,
        step=10
    )
    
    st.divider()
    st.caption("Built with LangChain + Streamlit")

# Main content
st.markdown('<p class="main-header">📄 Document Analyzer Agent</p>', unsafe_allow_html=True)

# Create tabs for different sections
tab1, tab2, tab3 = st.tabs(["💬 Query", "📊 Index Info", "⚙️ Manage Index"])

# ===== TAB 1: Query =====
with tab1:
    st.markdown('<p class="section-header">Ask Questions About Your Document</p>', unsafe_allow_html=True)
    
    col1, col2 = st.columns([0.8, 0.2])
    
    with col1:
        query = st.text_input(
            "Enter your question:",
            placeholder="What is this document about?",
            label_visibility="collapsed"
        )
    
    with col2:
        search_button = st.button("🔍 Search", use_container_width=True)
    
    if search_button and query:
        try:
            with st.spinner("Loading index and generating answer..."):
                # Load embeddings and vectorstore
                embeddings_model = OpenAIEmbeddings()
                vectorstore = FAISS.load_local(
                    FAISS_INDEX_FILE,
                    embeddings_model,
                    allow_dangerous_deserialization=True
                )
                
                # Create QA chain
                qa = RetrievalQA.from_chain_type(
                    llm=ChatOpenAI(
                        model_name=model_name,
                        temperature=temperature
                    ),
                    retriever=vectorstore.as_retriever(
                        search_kwargs={"k": num_docs}
                    )
                )
                
                # Get answer
                answer = qa.run(query)
                
                # Display answer
                st.markdown('<div class="answer-box">', unsafe_allow_html=True)
                st.markdown(f"**Answer:**\n\n{answer}")
                st.markdown('</div>', unsafe_allow_html=True)
                
                # Retrieve and display metadata
                st.markdown('<p class="section-header">📍 Retrieved Chunks</p>', unsafe_allow_html=True)
                retriever = vectorstore.as_retriever(
                    search_kwargs={"k": num_docs}
                )
                retrieved_docs = retriever.invoke(query)
                
                for idx, doc in enumerate(retrieved_docs, 1):
                    with st.expander(f"📄 Chunk {idx}", expanded=(idx == 1)):
                        st.text(doc.page_content)
                        if doc.metadata:
                            st.caption(f"**Metadata:** {doc.metadata}")
        
        except FileNotFoundError:
            st.error("⚠️ FAISS index not found. Please build the index first in the 'Manage Index' tab.")
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
    
    elif search_button:
        st.warning("⚠️ Please enter a question.")

# ===== TAB 2: Index Info =====
with tab2:
    st.markdown('<p class="section-header">📊 Index Information</p>', unsafe_allow_html=True)
    
    try:
        # Load index info
        embeddings_model = OpenAIEmbeddings()
        vectorstore = FAISS.load_local(
            FAISS_INDEX_FILE,
            embeddings_model,
            allow_dangerous_deserialization=True
        )
        
        # Get index stats
        index_path = Path(FAISS_INDEX_FILE)
        index_files = list(index_path.glob("*")) if index_path.exists() else []
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Index Status", "✅ Active")
        with col2:
            st.metric("Index Files", len(index_files))
        with col3:
            if index_path.exists():
                size_mb = sum(f.stat().st_size for f in index_files) / (1024 * 1024)
                st.metric("Index Size", f"{size_mb:.2f} MB")
        
        st.divider()
        
        # Document info
        st.subheader("📄 Document Information")
        
        # Check sample data
        sample_dir = Path(SAMPLE_DATA_DIR)
        if sample_dir.exists():
            sample_files = list(sample_dir.glob("*.txt"))
            st.write(f"**Sample Files:** {len(sample_files)} file(s)")
            
            for file in sample_files:
                with open(file, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    col1, col2 = st.columns(2)
                    with col1:
                        st.write(f"**File:** {file.name}")
                    with col2:
                        st.write(f"**Size:** {len(content)} characters")
        
        st.divider()
        
        # Index settings used
        st.subheader("⚙️ Current Settings")
        settings_col1, settings_col2 = st.columns(2)
        with settings_col1:
            st.write(f"**Chunk Size:** {chunk_size}")
            st.write(f"**Model:** {model_name}")
        with settings_col2:
            st.write(f"**Chunk Overlap:** {chunk_overlap}")
            st.write(f"**Temperature:** {temperature}")
    
    except FileNotFoundError:
        st.info("ℹ️ No FAISS index found yet. Build it first in the 'Manage Index' tab.")

# ===== TAB 3: Manage Index =====
with tab3:
    st.markdown('<p class="section-header">⚙️ Manage FAISS Index</p>', unsafe_allow_html=True)
    
    # Subtabs for upload and index management
    mgmt_tab1, mgmt_tab2 = st.tabs(["📤 Upload Document", "🔧 Manage Index"])
    
    # Upload Documents Subtab
    with mgmt_tab1:
        st.subheader("📤 Upload Your Document")
        st.write("Upload a text file to create a new FAISS index.")
        
        uploaded_file = st.file_uploader(
            "Choose a text file",
            type=["txt"],
            label_visibility="collapsed"
        )
        
        if uploaded_file is not None:
            # Save uploaded file to sample_data
            try:
                sample_dir = Path(SAMPLE_DATA_DIR)
                sample_dir.mkdir(exist_ok=True)
                
                file_path = sample_dir / uploaded_file.name
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                st.success(f"✅ File uploaded: {uploaded_file.name}")
                st.info(f"File saved to: {file_path}")
                
                # Show file preview
                with st.expander("👁️ Preview", expanded=False):
                    content = file_path.read_text(encoding='utf-8', errors='ignore')
                    preview_lines = content.split('\n')[:20]
                    st.text('\n'.join(preview_lines))
                    if len(content.split('\n')) > 20:
                        st.caption(f"... (showing first 20 lines of {len(content.split(chr(10)))} total)")
                
                # Build index button
                col_left, col_right = st.columns(2)
                with col_left:
                    if st.button("🔨 Build Index from This File", key="build_uploaded", use_container_width=True):
                        try:
                            with st.spinner("Building index from uploaded file..."):
                                # Load and split
                                loader = TextLoader(str(file_path))
                                docs = loader.load()
                                
                                splitter = CharacterTextSplitter(
                                    chunk_size=chunk_size,
                                    chunk_overlap=chunk_overlap
                                )
                                chunks = splitter.split_documents(docs)
                                
                                # Create embeddings and FAISS
                                embeddings_model = OpenAIEmbeddings()
                                vectorstore = FAISS.from_documents(chunks, embeddings_model)
                                
                                # Save index
                                vectorstore.save_local(FAISS_INDEX_FILE)
                                
                                st.success(f"✅ Index built from {uploaded_file.name}!")
                                st.write(f"- **Chunks created:** {len(chunks)}")
                                st.write(f"- **File:** {uploaded_file.name}")
                                st.write(f"- **Chunk size:** {chunk_size}")
                                st.write(f"- **Overlap:** {chunk_overlap}")
                        
                        except Exception as e:
                            st.error(f"❌ Error building index: {str(e)}")
                
                with col_right:
                    if st.button("🗂️ Set as Default", key="set_default", use_container_width=True):
                        st.info(f"✅ {uploaded_file.name} will be used for the next index build.")
            
            except Exception as e:
                st.error(f"❌ Error uploading file: {str(e)}")
        
        st.divider()
        
        # Show uploaded files
        st.subheader("📚 Available Documents")
        sample_dir = Path(SAMPLE_DATA_DIR)
        if sample_dir.exists():
            sample_files = list(sample_dir.glob("*.txt"))
            if sample_files:
                for file in sample_files:
                    col1, col2 = st.columns([0.8, 0.2])
                    with col1:
                        size_kb = file.stat().st_size / 1024
                        st.write(f"📄 **{file.name}** ({size_kb:.2f} KB)")
                    with col2:
                        if st.button("🗑️", key=f"del_{file.name}", help="Delete file"):
                            try:
                                file.unlink()
                                st.success(f"Deleted {file.name}")
                                st.rerun()
                            except Exception as e:
                                st.error(f"Error: {str(e)}")
            else:
                st.info("No documents uploaded yet.")
        else:
            st.info("No sample_data directory found.")
    
    # Manage Index Subtab
    with mgmt_tab2:
        st.write("Build or rebuild the FAISS index from your documents.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🔨 Build Index", use_container_width=True):
                try:
                    with st.spinner("Building index... This may take a moment."):
                        # Load document
                        sample_file = Path(SAMPLE_DATA_DIR) / "sample.txt"
                        
                        if not sample_file.exists():
                            st.error(f"❌ Sample file not found: {sample_file}")
                            st.stop()
                        
                        # Load and split
                        loader = TextLoader(str(sample_file))
                        docs = loader.load()
                        
                        splitter = CharacterTextSplitter(
                            chunk_size=chunk_size,
                            chunk_overlap=chunk_overlap
                        )
                        chunks = splitter.split_documents(docs)
                        
                        # Create embeddings and FAISS
                        embeddings_model = OpenAIEmbeddings()
                        vectorstore = FAISS.from_documents(chunks, embeddings_model)
                        
                        # Save index
                        vectorstore.save_local(FAISS_INDEX_FILE)
                        
                        st.success(f"✅ Index built successfully!")
                        st.write(f"- **Chunks created:** {len(chunks)}")
                        st.write(f"- **Chunk size:** {chunk_size}")
                        st.write(f"- **Overlap:** {chunk_overlap}")
                
                except Exception as e:
                    st.error(f"❌ Error building index: {str(e)}")
        
        with col2:
            if st.button("🗑️ Delete Index", use_container_width=True):
                try:
                    import shutil
                    index_path = Path(FAISS_INDEX_FILE)
                    if index_path.exists():
                        shutil.rmtree(index_path)
                        st.success("✅ Index deleted successfully!")
                    else:
                        st.info("ℹ️ No index to delete.")
                except Exception as e:
                    st.error(f"❌ Error deleting index: {str(e)}")
        
        st.divider()
        
        # Index status
        st.subheader("📊 Index Status")
        
        index_path = Path(FAISS_INDEX_FILE)
        if index_path.exists():
            st.success("✅ Index exists and is ready to use")
            
            # Show index details
            index_files = list(index_path.glob("*"))
            with st.expander("📁 Index Files"):
                for file in index_files:
                    size_kb = file.stat().st_size / 1024
                    st.write(f"- {file.name}: {size_kb:.2f} KB")
        else:
            st.warning("⚠️ No index found. Build one to get started!")

st.divider()
st.caption("🚀 Document Analyzer Agent | Powered by LangChain + OpenAI + Streamlit")
