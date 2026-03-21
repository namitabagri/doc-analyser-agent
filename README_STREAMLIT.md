# 📄 Document Analyzer Agent - Streamlit UI

A modern web-based interface for the Document Analyzer Agent that enables semantic search and question-answering on your documents using LangChain, OpenAI, and Streamlit.

## ✨ Features

- **💬 Interactive Q&A**: Ask questions about your documents and get AI-powered answers
- **� Document Upload**: Upload `.txt` files directly through the UI
- **�📊 Index Management**: Build, rebuild, or delete the FAISS vector index
- **🔍 Retrieved Chunks Display**: See the exact chunks used to generate answers
- **⚙️ Customizable Settings**:
  - Multiple LLM models (GPT-4o-mini, GPT-4, GPT-3.5-turbo)
  - Adjustable temperature for creativity levels
  - Configurable chunk size and overlap
  - Control number of retrieved documents
- **📈 Index Information**: View statistics about your FAISS index
- **🎨 Beautiful UI**: Modern, responsive design with intuitive navigation

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- OpenAI API key (set as `OPENAI_API_KEY` environment variable)

### Installation

1. **Clone/navigate to the project directory:**
   ```bash
   cd /Users/namitabagri/Projects/march/doc-retrieval-system/doc-analyser-agent
   ```

2. **Create a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On macOS/Linux
   # or
   venv\Scripts\activate  # On Windows
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables:**
   Create a `.env` file in the project root:
   ```
   OPENAI_API_KEY=your-api-key-here
   ```

### Running the App

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## 📖 How to Use

### 1️⃣ **First Time Setup: Upload & Build Index**

**Option A: Upload Your Own Document**
- Go to the **⚙️ Manage Index** tab
- Click on **📤 Upload Document** subtab
- Click the upload area and select a `.txt` file
- Click **🔨 Build Index from This File**
- View the preview and statistics of your uploaded document

**Option B: Use Sample Document**
- Go to the **⚙️ Manage Index** tab
- Click on **🔧 Manage Index** subtab
- Click **🔨 Build Index** to use the default sample.txt
- Wait for the index to be built and saved

### 2️⃣ **Ask Questions**
- Go to the **💬 Query** tab
- Type your question in the search box
- Click **🔍 Search**
- View the AI-generated answer and retrieved chunks

### 3️⃣ **Monitor Index**
- Go to the **📊 Index Info** tab
- View index statistics and document information
- See current configuration

## 📁 Project Structure

```
doc-analyser-agent/
├── app.py                    # Main Streamlit app
├── main.py                   # Original CLI script
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── sample_data/
│   └── sample.txt           # Sample document
├── faiss_index/             # FAISS vector index (auto-generated)
└── .env                      # Environment variables (not in repo)
```

## ⚙️ Configuration

### Settings (Sidebar)
- **Model**: Choose between GPT-4o-mini, GPT-4, or GPT-3.5-turbo
- **Temperature**: 0.0 (focused) to 1.0 (creative)
- **Retrieved Chunks**: Number of chunks to use for answer generation
- **Chunk Size**: Characters per chunk (100-2000)
- **Chunk Overlap**: Overlap between chunks (0-500)

## 🔧 Troubleshooting

### Index not found
**Solution**: Build the index first in the "Manage Index" tab

### "Could not import faiss"
**Solution**: Install faiss-cpu: `pip install faiss-cpu`

### OpenAI API key errors
**Solution**: Ensure `OPENAI_API_KEY` is set in `.env` or environment variables

### Slow responses
**Solution**: Try reducing chunk size or using a faster model (GPT-3.5-turbo)

## 📝 Sample Queries

Try these questions with the sample document:
- "What is the document about?"
- "Who is mentioned in the document?"
- "What are the key achievements?"
- "Provide a summary of the content"

## 🛠️ Advanced Usage

### Using with Different Documents

**Method 1: Upload via UI (Recommended)**
1. Go to **⚙️ Manage Index** → **📤 Upload Document**
2. Click to upload your `.txt` file
3. Preview the file content
4. Click **🔨 Build Index from This File**
5. The index is built immediately and ready to use

**Method 2: Manual File Addition**
1. Add your `.txt` file to the `sample_data/` directory
2. Update line 28 in `app.py` to point to your file:
   ```python
   sample_file = Path(SAMPLE_DATA_DIR) / "your_file.txt"
   ```
3. Delete the old index in the "Manage Index" tab
4. Build a new index

**Upload Limitations:**
- Supports `.txt` files only
- Maximum file size depends on OpenAI embedding limits
- Files are saved to `sample_data/` directory
- You can delete files using the delete button (🗑️) in the upload tab

### Custom Chunk Settings

Modify the default values in the sidebar settings before building the index:
- **Smaller chunks** (200-300): Better for specific queries
- **Larger chunks** (800-1000): Better for maintaining context

## 📦 Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| streamlit | 1.28.1 | Web UI framework |
| langchain | 1.2.10 | LLM orchestration |
| langchain-community | 0.4.1 | Community integrations |
| langchain-classic | 1.0.3 | Legacy chain support |
| langchain-openai | 0.1.8 | OpenAI integration |
| langchain-text-splitters | 1.1.1 | Document splitting |
| faiss-cpu | 1.13.2 | Vector search |
| python-dotenv | 1.0.0 | Environment variables |

## 🎯 Key Improvements Over CLI

| Feature | CLI | Streamlit UI |
|---------|-----|-------------|
| User Interface | Terminal-only | Modern web UI |
| Multiple queries | One query per run | Unlimited queries |
| Settings adjustment | Code editing | Sidebar controls |
| Visual feedback | Text output | Rich UI elements |
| Index management | Manual | Built-in tools |
| Chunk visualization | None | Expandable boxes |
| Mobile friendly | No | Yes |

## 📊 How It Works

```
Document → TextLoader → Chunks → OpenAI Embeddings
                                        ↓
                                    FAISS Index
                                        ↓
User Query → Embedding → Search → Retrieved Chunks → ChatOpenAI → Answer
```

## 🔒 Security Notes

- Keep your `.env` file private (add to `.gitignore`)
- FAISS index contains serialized data - only load from trusted sources
- Use appropriate models based on your security requirements

## 📝 License

This project is part of the doc-retrieval-system

## 🤝 Contributing

Feel free to submit issues or pull requests for improvements!

## 💡 Tips & Tricks

1. **Speed up responses**: Use GPT-3.5-turbo instead of GPT-4
2. **Better accuracy**: Use GPT-4 with moderate temperature (0.3-0.5)
3. **Semantic search**: The index works best with 3-5 retrieved chunks
4. **Fresh index**: Rebuild if you update your documents
5. **Monitor usage**: Keep track of OpenAI API usage in your dashboard

---

**Built with ❤️ using LangChain + OpenAI + Streamlit**
