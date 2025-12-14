from sentence_transformers import SentenceTransformer

import numpy as np
import os
import json

class SemanticSearch:

    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embeddings = None
        self.documents = None
        self.document_map = {}

    def generate_embedding(self, text):
        if len(text.split()) == 0:
            raise ValueError("Empty Text")
        embedding = self.model.encode([text])[0]
        return embedding

    def build_embeddings(self, documents):
        self.documents = documents
        for each in self.documents:
            self.document_map[each["id"]] = each

        texts = [f"{doc['title']}: {doc['description']}" for doc in self.documents]
        self.embeddings = self.model.encode(texts, show_progress_bar=True)
        np.save("cache/movie_embeddings.npy", self.embeddings)
        return self.embeddings

    def load_or_create_embeddings(self, documents):
        self.documents = documents
        for each in self.documents:
            self.document_map[each["id"]] = each
        
        if os.path.exists("cache/movie_embeddings.npy"):
            self.embeddings = np.load("cache/movie_embeddings.npy")
            if len(self.embeddings) == len(documents):
                return self.embeddings
        return self.build_embeddings(documents)

    def search(self, query, limit=5):
        if self.embeddings is None:
            raise ValueError("No embeddings loaded. Call `load_or_create_embeddings` first.")
        final_list = []
        query_embed = self.generate_embedding(query)
        for i in range(len(self.embeddings)):
            cosine_scr = cosine_similarity(self.embeddings[i], query_embed)
            mov_iden = self.documents[i]
            final_list.append((cosine_scr, mov_iden))

        final_list.sort(key=lambda x: x[0], reverse=True)
        return final_list[:limit]




semantic_instance = SemanticSearch()

def verify_embeddings():
    path = os.path.join(os.path.dirname(__file__), "../data/movies.json")
    with open(path, "r") as f:
        movies_data = json.load(f)
    documents = movies_data["movies"]
    result = semantic_instance.load_or_create_embeddings(documents)
    print(f"Number of docs:   {len(documents)}")
    print(f"Embeddings shape: {result.shape[0]} vectors in {result.shape[1]} dimensions")   

def verify_model():
    print(f"Model loaded: {semantic_instance.model}")
    print(f"Max sequence length: {semantic_instance.model.max_seq_length}")

def embed_text(text):
    result = semantic_instance.generate_embedding(text)
    print(f"Text: {text}")
    print(f"First 3 dimensions: {result[:3]}")
    print(f"Dimensions: {result.shape[0]}")

def embed_query_text(query):
    result = semantic_instance.generate_embedding(query)
    print(f"Query: {query}")
    print(f"First 5 dimensions: {result[:5]}")
    print(f"Shape: {result.shape}")

def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)

