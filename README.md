# doc-analyser-agent
Implementation of an agentic RAG system using LLMs, document retrieval, and tool-based decision making.

---------------------------- Document Ingestion --------------------------------------

Document
   ↓
Split into chunks
   ↓
Convert chunks → embeddings
   ↓
Store in vector database

---------------------------- Query Flow -----------------------------------------------

User Question
     ↓
Agent (decision-maker) [Decides based on prompt, rules, tool descriptions]
     ↓
 ├── Use LLM directly
 │        ↓
 │     LLM generates answer
 │
 └── OR use Document Search Tool (RAG):
           ↓
     Convert question → embedding
           ↓
     Search vector database
           ↓
     Retrieve top-K relevant chunks
           ↓
     Send (question + context) → LLM
           ↓
     LLM generates answer

     ↓
Final Answer