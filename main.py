from doc_loader import load_document
from text_splitter import split_text
from embeddings import get_embedding
from retrieval import find_most_similar
from openai import OpenAI
import os
from dotenv import load_dotenv

# Load env + client
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Load + split
doc = load_document("sample_data/sample.txt")
chunks = split_text(doc)

# Embed chunks
chunk_embeddings = [get_embedding(chunk) for chunk in chunks]

# User query
query = input("Ask a question: ")

# Embed query
query_embedding = get_embedding(query)

# Retrieve best chunk
best_index, _ = find_most_similar(query_embedding, chunk_embeddings)
context = chunks[best_index]

#  Send to LLM
response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {
            "role": "system",
            "content": "Answer the question using the provided context. Be concise."
        },
        {
            "role": "user",
            "content": f"Context:\n{context}\n\nQuestion:\n{query}"
        }
    ]
)

print("\nAnswer:\n")
print(response.choices[0].message.content)