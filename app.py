import os
from dotenv import load_dotenv
from flask import Flask, render_template, request, jsonify
from sentence_transformers import SentenceTransformer
from googleapiclient.discovery import build
import numpy as np
import config

load_dotenv()

app = Flask(__name__)

# Initialize models and services
model = SentenceTransformer(config.Config.EMBEDDING_MODEL)
google_api_key = os.getenv('GOOGLE_API_KEY')
google_cse_id = os.getenv('GOOGLE_CSE_ID')
search_service = build("customsearch", "v1", developerKey=google_api_key)

def get_google_results(query, num_results=10):
    try:
        res = search_service.cse().list(
            q=query,
            cx=google_cse_id,
            num=num_results
        ).execute()
        return res.get('items', [])
    except Exception as e:
        print(f"Error fetching Google results: {e}")
        return []

def calculate_similarity(query_embedding, results):
    # Extract snippets or titles for comparison
    texts = [f"{item.get('title', '')} {item.get('snippet', '')}" for item in results]
    text_embeddings = model.encode(texts, convert_to_tensor=True)
    
    # Calculate cosine similarity
    similarities = np.inner(query_embedding, text_embeddings)
    
    # Pair results with similarity scores
    scored_results = []
    for i, result in enumerate(results):
        scored_results.append({
            **result,
            'similarity': float(similarities[i])
        })
    
    # Sort by similarity
    scored_results.sort(key=lambda x: x['similarity'], reverse=True)
    return scored_results

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        query = request.form.get('query')
        if not query:
            return render_template('index.html', error="Please enter a search query")
        
        # Get query embedding
        query_embedding = model.encode(query, convert_to_tensor=True)
        
        # Get initial results from Google
        google_results = get_google_results(query)
        
        if not google_results:
            return render_template('index.html', error="No results found")
        
        # Re-rank by semantic similarity
        ranked_results = calculate_similarity(query_embedding, google_results)
        
        return render_template('index.html', 
                             query=query, 
                             results=ranked_results[:config.Config.RESULTS_PER_PAGE])
    
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)