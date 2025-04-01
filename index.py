from flask import Flask, request, jsonify
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
import faiss
import numpy as np
import json
from dotenv import load_dotenv
import os
from flask_cors import CORS


app = Flask(__name__)



CORS(app)
load_dotenv()

# Initialize Groq API
# GROQ_API_KEY = os.getenv("GROQ_API_KEY") 
GROQ_API_KEY ="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" # Replace with your actual API key

llm = ChatGroq(model_name="llama-3.3-70b-versatile", api_key=GROQ_API_KEY,temperature=1)

# Load stored website embeddings
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
index = faiss.read_index("faiss_index.index")
with open("website_data.json", "r", encoding="utf-8") as f:
    website_texts = json.load(f)

@app.route("/test", methods=["POST"])
def your_function():
    print("request received")
    return {"message": "Success"}

@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.get_json()
        print(data)
        user_query = data.get("query")
        

        if not user_query:
            return jsonify({"error": "Query is required"}), 400

        # Convert user query to embedding
        user_embedding = embedding_model.embed_query(user_query)
        user_embedding = np.array(user_embedding).reshape(1, -1)

        # Search for the closest matching website content
        _, idx = index.search(user_embedding, 1)
        matched_text = website_texts[idx[0][0]]

        # Combine the matched content with the user query for LLM refinement
        final_prompt = f"""
        User Query: {user_query}
        Relevant Website Content: {matched_text}
        
        Provide a **concise response** (max 50 words). Be **short and precise**.
        """

        # Get response from Llama3
        response = llm.invoke(final_prompt, max_tokens=100)

        if hasattr(response, "content"):  # Check if response has 'content' attribute
            response_text = response.content
        else:
            response_text = str(response)  # Convert to string if needed

        return jsonify({
            "query": user_query,
            "relevant_content": matched_text,
            "response": response_text
        })

    except Exception as e:
        import traceback
        print(traceback.format_exc())  # Logs full error in terminal
        return jsonify({"error": f"Internal Server Error: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
