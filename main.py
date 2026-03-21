from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_classic.chains import RetrievalQA
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# File path for FAISS index
FAISS_INDEX_FILE = "faiss_index"

# 1️⃣ Load your document
loader = TextLoader("sample_data/sample.txt")
docs = loader.load()

# 2️⃣ Split into chunks
splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = splitter.split_documents(docs)

# 3️⃣ Create embeddings + FAISS vector store
vectorstore = FAISS.from_documents(chunks, OpenAIEmbeddings())

# 4️⃣ Save the index to disk
vectorstore.save_local(FAISS_INDEX_FILE)
print("FAISS index saved to disk ✅")

# 5️⃣ Later or on restart: Load index from disk
vectorstore = FAISS.load_local(FAISS_INDEX_FILE, OpenAIEmbeddings(), allow_dangerous_deserialization=True)
print("FAISS index loaded from disk ✅")

# 6️⃣ Create RAG chain
qa = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model_name="gpt-4o-mini", temperature=0),
    retriever=vectorstore.as_retriever()
)

# 7️⃣ Ask a question
query = input("Ask a question: ")
answer = qa.run(query)
print("\nAnswer:\n", answer)