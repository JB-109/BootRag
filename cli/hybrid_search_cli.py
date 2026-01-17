import argparse
import os
import json

from hybrid_search import HybridSearch
from dotenv import load_dotenv
from google import genai
from sentence_transformers import CrossEncoder

parser = argparse.ArgumentParser(description="Hybrid Search CLI")
subparsers = parser.add_subparsers(dest="command", help="Available commands")

weighted_search_parser = subparsers.add_parser("weighted-search", help="Perform weighted hybrid search")
weighted_search_parser.add_argument("query", type=str, help="Search query")
weighted_search_parser.add_argument("--alpha", type=float, default=0.5, help="Weight for BM25 vs semantic (default: 0.5)")
weighted_search_parser.add_argument("--limit", type=int, default=5, help="Number of results to return (default: 5)")

rrf_search_parser = subparsers.add_parser("rrf-search", help="Perform RRF hybrid search")
rrf_search_parser.add_argument("query", type=str, help="Search query")
rrf_search_parser.add_argument("-k", type=int, default=60, help="RRF k parameter (default: 60)")
rrf_search_parser.add_argument("--limit", type=int, default=5, help="Number of results to return (default: 5)")
rrf_search_parser.add_argument("--enhance", type=str, choices=["spell", "rewrite", "expand"], help="Query enhancement method")
rrf_search_parser.add_argument("--rerank-method", type=str, choices=["individual", "batch", "cross_encoder"], help="Reranking method")

args = parser.parse_args()


def main() -> None:
    
    match args.command:
        
        case "weighted-search":

            # gets the movies.
            path = os.path.join(os.path.dirname(__file__), "../data/movies.json")
            with open(path, "r") as f:
                movies_data = json.load(f)
            documents = movies_data["movies"]
            
            # Perform hybrid search
            hybrid_search = HybridSearch(documents)
            results = hybrid_search.weighted_search(args.query, args.alpha, args.limit)
            
            # Print results
            for i, result in enumerate(results, 1):
                print(f"{i}. {result['title']}")
                print(f"   Hybrid Score: {result['hybrid_score']:.3f}")
                print(f"   BM25: {result['bm25_score']:.3f}, Semantic: {result['semantic_score']:.3f}")
                print(f"   {result['document']}...")
        
        case "rrf-search":
            
            # gets the movies.
            path = os.path.join(os.path.dirname(__file__), "../data/movies.json")
            with open(path, "r") as f:
                movies_data = json.load(f)
            documents = movies_data["movies"]
            
            # Handle query enhancement
            # this one checks for the lexical typos via the LLM.
            query = args.query
            if args.enhance == "spell":
                load_dotenv()
                client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))
                
                prompt = f"""Fix any spelling errors in this movie search query.
                    Only correct obvious typos. Don't change correctly spelled words.
                    Query: "{query}"
                    If no errors, return the original query.
                    Corrected:"""
                                    
                response = client.models.generate_content(
                    model="gemini-2.5-flash-lite",
                    contents=prompt
                )
                
                enhanced_query = response.text.strip()
                print(f"Enhanced query ({args.enhance}): '{query}' -> '{enhanced_query}'\n")
                query = enhanced_query


            elif args.enhance == "rewrite":
                load_dotenv()
                api_key = os.environ.get("GEMINI_API_KEY")
                client = genai.Client(api_key=api_key)
                
                prompt = f"""Rewrite this movie search query to be more specific and searchable.
                    Original: "{query}"
                    Consider:
                    - Common movie knowledge (famous actors, popular films)
                    - Genre conventions (horror = scary, animation = cartoon)
                    - Keep it concise (under 10 words)

                    Examples:
                    - "that bear movie where leo gets attacked" -> "The Revenant Leonardo DiCaprio bear attack"
                    - "movie about bear in london with marmalade" -> "Paddington London marmalade"
                    - "scary movie with bear from few years ago" -> "bear horror movie 2015-2020"
                    Rewritten query:"""
                
                response = client.models.generate_content(
                    model="gemini-2.5-flash-lite",
                    contents=prompt
                )
                
                enhanced_query = response.text.strip()
                print(f"Enhanced query ({args.enhance}): '{query}' -> '{enhanced_query}'\n")
                query = enhanced_query


            elif args.enhance == "expand":
                load_dotenv()
                api_key = os.environ.get("GEMINI_API_KEY")
                client = genai.Client(api_key=api_key)
                
                prompt = f"""Expand this movie search query with related terms.
                    Add synonyms and related concepts that might appear in movie descriptions.
                    Keep expansions relevant and focused.
                    This will be appended to the original query.

                    Examples:

                    - "scary bear movie" -> "scary horror grizzly bear movie terrifying film"
                    - "action movie with bear" -> "action thriller bear chase fight adventure"
                    - "comedy with bear" -> "comedy funny bear humor lighthearted"
                    Query: "{query}"
                    """
                
                response = client.models.generate_content(
                    model="gemini-2.5-flash-lite",
                    contents=prompt
                )
                
                enhanced_query = response.text.strip()
                print(f"Enhanced query ({args.enhance}): '{query}' -> '{enhanced_query}'\n")
                query = enhanced_query
            
            # Perform RRF hybrid search
            hybrid_search = HybridSearch(documents)
            
            # Determine how many results to fetch
            # It needs a larger pool of candidates for reranking.
            fetch_limit = args.limit * 5 if args.rerank_method in ["individual", "batch", "cross_encoder"] else args.limit
            results = hybrid_search.rrf_search(query, args.k, fetch_limit)
            
            # Handle re-ranking
            if args.rerank_method == "individual":

                print(f"Reranking top {args.limit} results using individual method...")
                load_dotenv()
                api_key = os.environ.get("GEMINI_API_KEY")
                client = genai.Client(api_key=api_key)
                
                # Build a single prompt with all documents
                movies_list = ""
                for idx, result in enumerate(results, 1):
                    movies_list += f"{idx}. {result.get('title', '')} - {result.get('document', '')}\n\n"
                
                prompt = f"""Rate how well each of these movies matches the search query.
                    Query: "{query}"
                    Movies:
                    {movies_list}

                    Consider for each movie:
                    - Direct relevance to query
                    - User intent (what they're looking for)
                    - Content appropriateness

                    Rate each movie 0-10 (10 = perfect match).
                    Respond with ONLY the scores in order, one per line, no other text.

                    Example format:
                    8.5
                    7.0
                    9.5
                    Scores:"""
                

                response = client.models.generate_content(
                    model="gemini-2.5-flash-lite",
                    contents=prompt
                )
                
                # Parse scores from response
                # it updates the existing results list of dict with the new rerank_score key.
                scores_text = response.text.strip().split('\n')
                for idx, result in enumerate(results):
                    try:
                        if idx < len(scores_text):
                            score = float(scores_text[idx].strip())
                        else:
                            score = 0.0
                    except (ValueError, IndexError):
                        score = 0.0
                    
                    result['rerank_score'] = score
                
                results = sorted(results, key=lambda x: x['rerank_score'], reverse=True)[:args.limit]
                
                print(f"Reciprocal Rank Fusion Results for '{query}' (k={args.k}):\n")
                
                for i, result in enumerate(results, 1):
                    print(f"{i}. {result['title']}")
                    print(f"   Rerank Score: {result['rerank_score']:.3f}/10")
                    print(f"   RRF Score: {result['rrf_score']:.3f}")
                    bm25_rank = result['bm25_rank'] if result['bm25_rank'] else "N/A"
                    semantic_rank = result['semantic_rank'] if result['semantic_rank'] else "N/A"
                    print(f"   BM25 Rank: {bm25_rank}, Semantic Rank: {semantic_rank}")
                    print(f"   {result['document']}...")

            elif args.rerank_method == "batch":

                print(f"Reranking top {args.limit} results using batch method...\n")

                load_dotenv()
                api_key = os.environ.get("GEMINI_API_KEY")
                client = genai.Client(api_key=api_key)
                
                # Build document list with IDs
                doc_list_str = ""
                for idx, result in enumerate(results):
                    result['temp_id'] = idx
                    doc_list_str += f"{idx}. {result.get('title', '')} - {result.get('document', '')}\n\n"
                
                prompt = f"""Rank these movies by relevance to the search query.
                    Query: "{query}"
                    Movies:
                    {doc_list_str}
                    Return ONLY the IDs in order of relevance (best match first). Return a valid JSON list, nothing else. For example:
                    [75, 12, 34, 2, 1]
                    """
                
                response = client.models.generate_content(
                    model="gemini-2.5-flash-lite",
                    contents=prompt
                )
                
                # Parse JSON response
                # If the JSON parsing or LLM response fails,
                # ranked_ids is appended with index only,
                # In order to revert back the focus to the rrf ranking.
                try:
                    ranked_ids = json.loads(response.text.strip())
                except json.JSONDecodeError:
                    # Fallback: keep original order
                    ranked_ids = [i for i in range(len(results))]
                
                # Create a mapping of temp_id to rank
                # now the index is linked with the rank
                rank_map = {}
                for rank, temp_id in enumerate(ranked_ids, 1):
                    rank_map[temp_id] = rank
                
                # Assign rerank_rank to results
                # new key is added to the existing result list from the rrf search,
                # this key is assigned to its correspondin rank.
                for result in results:
                    result['rerank_rank'] = rank_map.get(result['temp_id'], 999)
                
                results = sorted(results, key=lambda x: x['rerank_rank'])[:args.limit]
                
                print(f"Reciprocal Rank Fusion Results for '{query}' (k={args.k}):")
                
                for i, result in enumerate(results, 1):
                    print(f"{i}. {result['title']}")
                    print(f"   Rerank Rank: {result['rerank_rank']}")
                    print(f"   RRF Score: {result['rrf_score']:.3f}")
                    bm25_rank = result['bm25_rank'] if result['bm25_rank'] else "N/A"
                    semantic_rank = result['semantic_rank'] if result['semantic_rank'] else "N/A"
                    print(f"   BM25 Rank: {bm25_rank}, Semantic Rank: {semantic_rank}")
                    print(f"   {result['document']}...")


            elif args.rerank_method == "cross_encoder":
                print(f"Reranking top {args.limit} results using cross_encoder method...\n")
                
                # Created pairs of [query, document]
                pairs = []
                for result in results:
                    doc_str = f"{result.get('title', '')} - {result.get('document', '')}"
                    pairs.append([query, doc_str])
                
                # Createed cross-encoder and compute scores
                cross_encoder = CrossEncoder("cross-encoder/ms-marco-TinyBERT-L2-v2")
                scores = cross_encoder.predict(pairs)
                
                # Assign scores to results
                for idx, result in enumerate(results):
                    result['cross_encoder_score'] = scores[idx]
                
                results = sorted(results, key=lambda x: x['cross_encoder_score'], reverse=True)[:args.limit]
                
                print(f"Reciprocal Rank Fusion Results for '{query}' (k={args.k}):")
                
                for i, result in enumerate(results, 1):
                    print(f"{i}. {result['title']}")
                    print(f"   Cross Encoder Score: {result['cross_encoder_score']:.3f}")
                    print(f"   RRF Score: {result['rrf_score']:.3f}")
                    bm25_rank = result['bm25_rank'] if result['bm25_rank'] else "N/A"
                    semantic_rank = result['semantic_rank'] if result['semantic_rank'] else "N/A"
                    print(f"   BM25 Rank: {bm25_rank}, Semantic Rank: {semantic_rank}")
                    print(f"   {result['document']}...")
            else:
                
                # Print results without re-ranking
                for i, result in enumerate(results, 1):
                    print(f"{i}. {result['title']}")
                    print(f"   RRF Score: {result['rrf_score']:.3f}")
                    bm25_rank = result['bm25_rank'] if result['bm25_rank'] else "N/A"
                    semantic_rank = result['semantic_rank'] if result['semantic_rank'] else "N/A"
                    print(f"   BM25 Rank: {bm25_rank}, Semantic Rank: {semantic_rank}")
                    print(f"   {result['document']}...")
        
        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
