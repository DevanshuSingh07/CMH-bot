import faiss
import numpy as np
import json
from langchain_huggingface import HuggingFaceEmbeddings

# Load website data
with open("faq_data.json", "r", encoding="utf-8") as f:
    website_json = json.load(f)

# Extract all questions from nested structure
text_data = []
for category in website_json.get("faqs", []):
    for q in category.get("questions", []):
        question_text = q.get("question")
        if question_text:
            text_data.append(question_text)

# Safety check
if not text_data:
    raise ValueError("❌ No questions found in faq_data.json. Check your JSON structure.")

# Initialize embedding model
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# Generate embeddings
website_embeddings = [embedding_model.embed_query(text) for text in text_data]
website_embeddings = np.array(website_embeddings, dtype=np.float32)

# Check embedding shape
if website_embeddings.ndim != 2 or website_embeddings.shape[0] == 0:
    raise ValueError("❌ Embedding generation failed or returned empty result.")

# Create and populate FAISS index
index = faiss.IndexFlatL2(website_embeddings.shape[1])
index.add(website_embeddings)

# Save the index
faiss.write_index(index, "faiss_index.index")

print("✅ FAISS index created and saved as 'faiss_index.index'")
