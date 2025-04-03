import os
import json
import numpy as np
import faiss
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from flask_cors import CORS
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

# Load environment variables
dotenv_path = os.path.join(os.path.dirname(__file__), ".env")
load_dotenv(dotenv_path)

# Secure API Key Retrieval
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    raise RuntimeError("Missing GROQ_API_KEY in environment variables.")

# Lazy-loaded models
llm = None
embedding_model = None
index = None
website_texts = None

def load_models():
    """Lazy load models only when needed and avoid reloading"""
    global llm, embedding_model, index, website_texts

    if llm is None:
        llm = ChatGroq(model_name="llama-3.3-70b-versatile", api_key=GROQ_API_KEY, temperature=1)

    if embedding_model is None:
        embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    if index is None and os.path.exists("faiss_index.index"):
        index = faiss.read_index("faiss_index.index")

    if website_texts is None and os.path.exists("website_data.json"):
        with open("website_data.json", "r", encoding="utf-8") as f:
            website_texts = json.load(f)

            
def create_app():
    """Creates Flask app for Gunicorn compatibility."""
    app = Flask(__name__)
    CORS(app)

    @app.route("/test", methods=["GET"])
    def test():
        return jsonify({"message": "Server is running"}), 200

    @app.route("/chat", methods=["POST"])
    def chat():
        """Handles user queries and generates responses."""
        try:
            data = request.get_json()
            user_query = data.get("query", "").strip()

            if not user_query:
                return jsonify({"error": "Query is required"}), 400

            # Load models if not already loaded
            load_models()

            # Convert query to embedding
            user_embedding = np.array(embedding_model.embed_query(user_query)).reshape(1, -1)

            # Find closest match in index
            _, idx = index.search(user_embedding, 1)
            matched_text = website_texts[idx[0][0]]

            # Create final prompt
            final_prompt = f"""
            User Query: {user_query}
            Relevant Website Content: {matched_text}
            Provide a **concise response** (max 50 words). Be **short and precise**.
            """

            # Get response from Llama3
            response = llm.invoke(final_prompt, max_tokens=100)

            response_text = response.content if hasattr(response, "content") else str(response)

            return jsonify({
                "query": user_query,
                "relevant_content": matched_text,
                "response": response_text
            })

        except Exception as e:
            import traceback
            print(traceback.format_exc())  # Log full error
            return jsonify({"error": f"Internal Server Error: {str(e)}"}), 500

    return app

# Gunicorn entry point
app = create_app()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
