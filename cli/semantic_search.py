from copyreg import pickle
from sentence_transformers import SentenceTransformer

import numpy as np
import os
import json
import re

class SemanticSearch:

    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.embeddings = None
        self.documents = None
        self.document_map = {}

    # it generates embedding for a single text.
    def generate_embedding(self, text):
        if len(text.split()) == 0:
            raise ValueError("Empty Text")
        embedding = self.model.encode([text])[0]
        return embedding

    # it generates embedding of the whole doc via batch processing.
    def build_embeddings(self, documents):
        self.documents = documents
        for each in self.documents:
            self.document_map[each["id"]] = each

        texts = [f"{doc['title']}: {doc['description']}" for doc in self.documents]
        self.embeddings = self.model.encode(texts, show_progress_bar=True)
        np.save("cache/movie_embeddings.npy", self.embeddings)
        return self.embeddings

    # it checks if the embeddings are computed and stored, and if stored, are they updated?
    # if they are, loaded in, and if not, then recomputed.
    def load_or_create_embeddings(self, documents):
        self.documents = documents
        for each in self.documents:
            self.document_map[each["id"]] = each
        
        if os.path.exists("cache/movie_embeddings.npy"):
            self.embeddings = np.load("cache/movie_embeddings.npy")
            if len(self.embeddings) == len(documents):
                return self.embeddings
        return self.build_embeddings(documents)

    # it embeds the query using the generate_embeddings and then ranks the movies based on their score.
    # calculated via cosine similarity func.
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


class ChunkedSemanticSearch(SemanticSearch):

    def __init__(self) -> None:
        super().__init__()
        self.chunk_embeddings = None
        self.chunk_metadata = None

    def semantic_chunk(self, text, max_chunk_size=4, overlap=1):
        """Split text into semantic chunks by sentences"""

        text = text.strip()
        if not text:
            return []
        
        # here regex is basically saying, match the whitespaces which are preceded by one of the char.
        # and split it from there.
        sentences = re.split(r"(?<=[.!?])\s+", text)
        
        # Handle single sentence without punctuation
        if len(sentences) == 1 and not re.search(r'[.!?]$', sentences[0]):
            return [text]
        
        sentences = [s.strip() for s in sentences if s.strip()]
        if not sentences:
            return []
        
        chunks = []
        
        if overlap > 0:
            i = 0
            while i < len(sentences):
                chunk_end = i + max_chunk_size
                chunk = " ".join(sentences[i:chunk_end])

                chunk = chunk.strip()
                if chunk:
                    chunks.append(chunk)
                
                i = i + max_chunk_size - overlap
                
                # Stop if we've reached or passed the end
                if chunk_end >= len(sentences):
                    break
                    
        else:
            for i in range(0, len(sentences), max_chunk_size):
                chunk = " ".join(sentences[i:i + max_chunk_size])

                chunk = chunk.strip()
                if chunk:
                    chunks.append(chunk)
        
        return chunks


    def build_chunk_embeddings(self, documents):
        """Build embeddings for document chunks"""
        self.documents = documents
        self.document_map = {}
        for doc in documents:
            self.document_map[doc["id"]] = doc
        
        all_chunks = []
        chunk_metadata = []

        # first each movie is fetched iteratively and then chunked.
        # then each chunk is added to a all_chunks list and then,
        # each movie id and chunk id is appeneded as a dict to a list.
        for doc_idx, doc in enumerate(documents):
            description = doc.get("description", "")
            if not description.strip():
                continue
            
            # chunk with 4 sentences, 1 sentence overlap
            chunks = self.semantic_chunk(description, max_chunk_size=4, overlap=1)
            
            for chunk_idx, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                chunk_metadata.append({
                    "movie_idx": doc_idx,
                    "chunk_idx": chunk_idx,
                    "total_chunks": len(chunks)
                })
        
        # Generate embeddings
        self.chunk_embeddings = self.model.encode(all_chunks, show_progress_bar=True)
        self.chunk_metadata = chunk_metadata
        
        # Save to cache
        os.makedirs("cache", exist_ok=True)
        np.save("cache/chunk_embeddings.npy", self.chunk_embeddings)
        
        with open("cache/chunk_metadata.json", "w") as f:
            json.dump(chunk_metadata, f, indent=2)
        
        return self.chunk_embeddings



    def load_or_create_chunk_embeddings(self, documents) -> np.ndarray:
        """Load cached chunk embeddings or create new ones"""

        self.documents = documents
        self.document_map = {}
        for doc in documents:
            self.document_map[doc["id"]] = doc

        
        if os.path.exists("cache/chunk_embeddings.npy") and os.path.exists("cache/chunk_metadata.json"):
            self.chunk_embeddings = np.load("cache/chunk_embeddings.npy")
            
            with open("cache/chunk_metadata.json", "r") as f:
                metadata = json.load(f)
                self.chunk_metadata = metadata
            
            return self.chunk_embeddings
        else:
            return self.build_chunk_embeddings(documents)



    def search_chunks(self, query, limit=10):
        """Search across chunk embeddings and aggregate results by document"""

        # availability of self.embeddings is checked,
        # embedding of query is created,
        # then score is calculated against each chunk,
        # then movies meta data is appended to a temp variable,
        # only movie id and score is appended to a another temp variable,
        # if higher score of a chunk is found for a movie id,
        # then the score is updated to the higher score,
        # and returned after sorting in decreasing order
        if self.chunk_embeddings is None:
            raise ValueError("No chunk embeddings loaded. call load_or_create_chunk_embeddings first.")
        
        # Generate query embedding
        query_embedding = self.generate_embedding(query)
        
        # Calculate similarity scores for all chunks
        chunk_scores = []
        for i, chunk_embedding in enumerate(self.chunk_embeddings):
            score = cosine_similarity(chunk_embedding, query_embedding)
            metadata = self.chunk_metadata[i]
            chunk_scores.append({
                "chunk_idx": metadata["chunk_idx"],
                "movie_idx": metadata["movie_idx"],
                "score": score
            })
        
        # Aggregate scores by movie (keep highest score per movie)
        movie_scores = {}
        for chunk_score in chunk_scores:
            movie_idx = chunk_score["movie_idx"]
            score = chunk_score["score"]
            
            # It repalces the score with the higher score for each movie index.
            if movie_idx not in movie_scores or score > movie_scores[movie_idx]:
                movie_scores[movie_idx] = score
        
        # Sort by score descending
        sorted_movies = sorted(movie_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Take top limit results
        top_movies = sorted_movies[:limit]
        
        # Format results
        results = []
        for movie_idx, score in top_movies:
            doc = self.documents[movie_idx]
            results.append({
                "id": doc["id"],
                "title": doc["title"],
                "document": doc.get("description", "")[:100],
                "score": round(score, 4),
            })
        
        return results



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
