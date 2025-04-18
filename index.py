import os
import json
import numpy as np
import faiss
import threading
import time
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from flask_cors import CORS
from concurrent.futures import ThreadPoolExecutor
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

# Load environment variables
dotenv_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(dotenv_path)
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError("Missing GROQ_API_KEY in environment variables.")

# Global resources
llm = None
embedding_model = None
index = None
website_texts = None
executor = ThreadPoolExecutor(max_workers=2)
faq_texts = None
faq_index = None

model_lock = threading.Lock()


def load_models():
    global llm, embedding_model, index, website_texts, faq_texts, faq_index

    start_time = time.time()
    with model_lock:
        if llm is None:
            print("Loading LLM...")
            llm = ChatGroq(model_name="llama-3.3-70b-versatile", api_key=GROQ_API_KEY, temperature=1)

        if embedding_model is None:
            print("Loading Embedding Model...")
            embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

        if faq_texts is None and os.path.exists("faq_data.json"):
            print("Loading and embedding FAQs...")
            with open("faq_data.json", "r", encoding="utf-8") as f:
                raw_faqs = json.load(f)
                flat_faqs = []
                for category in raw_faqs.get("faqs", []):
                    for q in category.get("questions", []):
                        flat_faqs.append(f"Q: {q['question']} A: {q['answer']}")
                faq_texts = flat_faqs

        if faq_index is None and faq_texts:
            vectors = embedding_model.embed_documents(faq_texts)
            faq_index = faiss.IndexFlatL2(len(vectors[0]))
            faq_index.add(np.array(vectors))

    print(f"✅ Models loaded in {time.time() - start_time:.2f} seconds.")


def handle_query(user_query):
    # Assume models are preloaded
    user_embedding = np.array(embedding_model.embed_query(user_query)).reshape(1, -1)

    _, idx = faq_index.search(user_embedding, 1)
    matched_text = faq_texts[int(idx[0][0])]

    final_prompt = f"""
    User Query: {user_query}
    Relevant Website Content: {matched_text}
    Provide a **concise response** (max 50 words). Be **short and precise**.
    """

    response = llm.invoke(final_prompt, max_tokens=100)
    response_text = response.content if hasattr(response, "content") else str(response)

    return {
        "query": user_query,
        "relevant_content": matched_text,
        "response": response_text
    }


def create_app():
    app = Flask(__name__)
    CORS(app)

    @app.route("/test", methods=["GET"])
    def test():
        return jsonify({"message": "Server is running"}), 200

    @app.route("/chat", methods=["POST"])
    def chat():
        try:
            data = request.get_json()
            user_query = data.get("query", "").strip()
            if not user_query:
                return jsonify({"error": "Query is required"}), 400

            future = executor.submit(handle_query, user_query)
            result = future.result(timeout=90)

            return jsonify(result)

        except Exception as e:
            import traceback
            print(traceback.format_exc())
            return jsonify({"error": f"Internal Server Error: {str(e)}"}), 500

    return app


# Create app and preload models once at startup
load_models()
app = create_app()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, threaded=True)
