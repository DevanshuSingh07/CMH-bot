import faiss
import numpy as np
import json
from langchain_huggingface import HuggingFaceEmbeddings

# Load website data
with open("website_data.json", "r", encoding="utf-8") as f:
    website_texts = json.load(f)

# Extract only text content
text_data = [entry["content"] for entry in website_texts if "content" in entry]

# Initialize embedding model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Generate embeddings for website content
website_embeddings = [embedding_model.embed_query(text) for text in text_data]
website_embeddings = np.array(website_embeddings, dtype=np.float32)

# Create FAISS index
index = faiss.IndexFlatL2(website_embeddings.shape[1])
index.add(website_embeddings)

# Save FAISS index
faiss.write_index(index, "faiss_index.index")

print("✅ FAISS index created and saved as 'faiss_index.index'")
