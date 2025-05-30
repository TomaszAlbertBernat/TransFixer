import os
import pickle
import argparse
import numpy as np
from tqdm import tqdm
import requests

# Configuration
OLLAMA_API_URL = "http://localhost:11434/api"
EMBEDDINGS_DIR = "embeddings"
DEFAULT_MODEL = "mxbai-embed-large:335m"
TOP_K = 5

def cosine_similarity(vec1, vec2):
    """Compute cosine similarity between two vectors."""
    dot_product = np.dot(vec1, vec2)
    norm_a = np.linalg.norm(vec1)
    norm_b = np.linalg.norm(vec2)
    
    if norm_a == 0 or norm_b == 0:
        return 0
    
    return dot_product / (norm_a * norm_b)

def get_embedding_from_ollama(query, model):
    """Get embeddings for the search query."""
    url = f"{OLLAMA_API_URL}/embeddings"
    payload = {
        "model": model,
        "prompt": query,
    }
    
    try:
        response = requests.post(url, json=payload)
        response.raise_for_status()
        return response.json()["embedding"]
    except (requests.exceptions.RequestException, KeyError) as e:
        print(f"Error getting embeddings: {str(e)}")
        return None

def load_embeddings(embeddings_dir):
    """Load all embeddings from the specified directory."""
    embeddings = []
    
    for root, _, files in os.walk(embeddings_dir):
        for file in files:
            if file.endswith('.pkl') and not file == "processing_summary.pkl":
                try:
                    file_path = os.path.join(root, file)
                    with open(file_path, 'rb') as f:
                        data = pickle.load(f)
                        embeddings.append(data)
                except Exception as e:
                    print(f"Error loading {file_path}: {str(e)}")
    
    print(f"Loaded {len(embeddings)} embeddings")
    return embeddings

def search_embeddings(query, model, embeddings_dir, top_k=TOP_K):
    """Search for transcriptions similar to the query."""
    # Get embedding for the query
    query_embedding = get_embedding_from_ollama(query, model)
    if query_embedding is None:
        return []
    
    # Load all embeddings
    all_embeddings = load_embeddings(embeddings_dir)
    
    # Calculate similarity with each embedding
    results = []
    for item in tqdm(all_embeddings, desc="Calculating similarities"):
        embedding = item["embedding"]
        metadata = item["metadata"]
        similarity = cosine_similarity(np.array(query_embedding), embedding)
        
        results.append({
            "similarity": similarity,
            "metadata": metadata,
            "source_file": metadata["source_file"]
        })
    
    # Sort by similarity (highest first)
    results.sort(key=lambda x: x["similarity"], reverse=True)
    
    # Return top k results
    return results[:top_k]

def extract_snippet(file_path, max_length=200):
    """Extract a snippet from the source file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
            
        if len(text) <= max_length:
            return text
        
        return text[:max_length] + "..."
    except Exception as e:
        return f"Error extracting snippet: {str(e)}"

def main():
    parser = argparse.ArgumentParser(description="Search through vectorized transcriptions")
    parser.add_argument("query", help="The search query")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"Ollama model to use (default: {DEFAULT_MODEL})")
    parser.add_argument("--embeddings_dir", default=EMBEDDINGS_DIR, help=f"Directory containing embeddings (default: {EMBEDDINGS_DIR})")
    parser.add_argument("--top_k", type=int, default=TOP_K, help=f"Number of results to return (default: {TOP_K})")
    args = parser.parse_args()
    
    # Search for similar transcriptions
    results = search_embeddings(args.query, args.model, args.embeddings_dir, args.top_k)
    
    if not results:
        print("No results found or error during search.")
        return
    
    # Display results
    print(f"\nTop {len(results)} results for query: '{args.query}'")
    print("-" * 80)
    
    for i, result in enumerate(results):
        print(f"{i+1}. Similarity: {result['similarity']:.4f}")
        print(f"   Source: {result['source_file']}")
        print(f"   Snippet: {extract_snippet(result['source_file'])}")
        print("-" * 80)

if __name__ == "__main__":
    main() 