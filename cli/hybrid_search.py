import os

from InvertedIndex import InvertedIndex
from semantic_search import ChunkedSemanticSearch


def normalize(scores):
    """Normalize scores using min-max normalization"""
    if not scores:
        return []
    
    min_score = min(scores)
    max_score = max(scores)
    
    # If all scores are the same, return all 1.0
    if min_score == max_score:
        return [1.0] * len(scores)
    
    # Apply min-max normalization
    return [(score - min_score) / (max_score - min_score) for score in scores]


class HybridSearch:
    def __init__(self, documents):
        self.documents = documents
        self.semantic_search = ChunkedSemanticSearch()
        self.semantic_search.load_or_create_chunk_embeddings(documents)
        self.idx = InvertedIndex()

        if not os.path.exists("cache/index.pkl"):
            self.idx.build({"movies": documents})
            self.idx.save()
        else:
            self.idx.load()

    def _bm25_search(self, query, limit):
        return self.idx.bm25_search(query, limit)

    def weighted_search(self, query, alpha, limit=5):
        """Perform weighted hybrid search combining BM25 and semantic scores"""

        # Get results from both searches (500x limit to ensure coverage)
        # It gets score of 500x the limit of movies from both searches.
        bm25_results = self._bm25_search(query, limit * 500)
        semantic_results = self.semantic_search.search_chunks(query, limit * 500)
        
        # Create dictionaries to store scores by document ID
        bm25_scores = {}
        semantic_scores = {}
        doc_map = {}
        
        # Process BM25 results
        # It appends all the returned movies into temp dict and docmap.
        for doc_id, score in bm25_results.items():
            bm25_scores[doc_id] = score
            doc_map[doc_id] = self.idx.docmap[doc_id]
        
        # Process semantic results
        # it does the same for sematic result,
        # and also finds the movie which arent added by the lexical search.
        for result in semantic_results:
            doc_id = result["id"]
            semantic_scores[doc_id] = result["score"]
            if doc_id not in doc_map:
                # Find the document
                for doc in self.documents:
                    if doc["id"] == doc_id:
                        doc_map[doc_id] = doc
                        break
        
        # Get all unique document IDs
        # it unionizes the two lists.
        all_doc_ids = set(bm25_scores.keys()) | set(semantic_scores.keys())
        
        # Normalize scores
        # It sets the score to 0.0 if a movies is not present in one of the search.
        # to normalize the score, each movie should be present in both lists.
        bm25_score_list = [bm25_scores.get(doc_id, 0.0) for doc_id in all_doc_ids]
        semantic_score_list = [semantic_scores.get(doc_id, 0.0) for doc_id in all_doc_ids]
        
        normalized_bm25 = normalize(bm25_score_list)
        normalized_semantic = normalize(semantic_score_list)
        
        # higher alpha means result biased towards keyword results.
        # lower alpha means lower biasness towards keyword based search.
        # Calculate hybrid scores
        results = []
        for i, doc_id in enumerate(all_doc_ids):
            bm25_norm = normalized_bm25[i]
            semantic_norm = normalized_semantic[i]
            hybrid_score = alpha * bm25_norm + (1 - alpha) * semantic_norm
            
            if doc_id in doc_map:
                results.append({
                    "id": doc_id,
                    "title": doc_map[doc_id]["title"],
                    "document": doc_map[doc_id].get("description", "")[:100],
                    "hybrid_score": hybrid_score,
                    "bm25_score": bm25_norm,
                    "semantic_score": semantic_norm
                })
        
        # Sort by hybrid score descending
        results.sort(key=lambda x: x["hybrid_score"], reverse=True)
        
        return results[:limit]


    def rrf_search(self, query, k, limit=10):
        """Perform RRF (Reciprocal Rank Fusion) hybrid search"""

        # Get results from both searches (500x limit)
        bm25_results = self._bm25_search(query, limit * 500)
        semantic_results = self.semantic_search.search_chunks(query, limit * 500)
        
        # Create dictionaries to store ranks and documents
        bm25_ranks = {}
        semantic_ranks = {}
        doc_map = {}
        
        # Process BM25 results (rank starts at 1)
        for rank, (doc_id, score) in enumerate(bm25_results.items(), 1):
            bm25_ranks[doc_id] = rank
            doc_map[doc_id] = self.idx.docmap[doc_id]
        
        # Process semantic results (rank starts at 1)
        for rank, result in enumerate(semantic_results, 1):
            doc_id = result["id"]
            semantic_ranks[doc_id] = rank
            if doc_id not in doc_map:
                # Find the document
                for doc in self.documents:
                    if doc["id"] == doc_id:
                        doc_map[doc_id] = doc
                        break
        
        # Get all unique document IDs
        all_doc_ids = set(bm25_ranks.keys()) | set(semantic_ranks.keys())
        
        # Calculate RRF scores
        results = []
        for doc_id in all_doc_ids:
            rrf_score = 0.0
            
            # higher k means flatter curves, not very sensitive to ranks,
            # whereas lower k means more aggresive to ranks, higher rank ranks higher.
            # Add BM25 RRF component if document appears in BM25 results
            if doc_id in bm25_ranks:
                rrf_score += 1.0 / (k + bm25_ranks[doc_id])
            
            # Add semantic RRF component if document appears in semantic results
            if doc_id in semantic_ranks:
                rrf_score += 1.0 / (k + semantic_ranks[doc_id])
            
            if doc_id in doc_map:
                results.append({
                    "id": doc_id,
                    "title": doc_map[doc_id]["title"],
                    "document": doc_map[doc_id].get("description", "")[:100],
                    "rrf_score": rrf_score,
                    "bm25_rank": bm25_ranks.get(doc_id, None),
                    "semantic_rank": semantic_ranks.get(doc_id, None)
                })
        
        # Sort by RRF score descending
        results.sort(key=lambda x: x["rrf_score"], reverse=True)
        
        return results[:limit]
