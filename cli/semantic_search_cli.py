from semantic_search import SemanticSearch
from semantic_search import embed_query_text
from semantic_search import embed_text
from semantic_search import verify_model
from semantic_search import verify_embeddings
from semantic_search import ChunkedSemanticSearch
import argparse
import os
import json


parser = argparse.ArgumentParser(description="Semantic Search CLI")
subparsers = parser.add_subparsers(dest="command", help="Available commands")

subparsers.add_parser("verify", help="Verify model loading")

embed_parser = subparsers.add_parser("embed_text", help="Generate embedding for text")
embed_parser.add_argument("text", type=str, help="Text to embed")

subparsers.add_parser("verify_embeddings", help="Verify embeddings loading")

embedquery_parser = subparsers.add_parser("embedquery", help="Generate embedding for query")
embedquery_parser.add_argument("embedquery", type=str, help="query to embed")

search_parser = subparsers.add_parser("search", help="Search for similar movies")
search_parser.add_argument("query", type=str, help="Search query")
search_parser.add_argument("--limit", type=int, default=5, help="Number of results to return (default: 5)")

chunk_parser = subparsers.add_parser("chunk", help="chunks the text")
chunk_parser.add_argument("chunk_text", type=str, help="text to chunk")
chunk_parser.add_argument("--chunk-size", type=int, nargs="?", default=200, help="chunk size")
chunk_parser.add_argument("--overlap", type=int, nargs="?", help="chunk size")

semantic_chunk_parser = subparsers.add_parser("semantic_chunk", help="Semantically chunk text")
semantic_chunk_parser.add_argument("text", type=str, help="Text to chunk")
semantic_chunk_parser.add_argument("--max-chunk-size", type=int, default=4, help="Maximum chunk size (default: 4)")
semantic_chunk_parser.add_argument("--overlap", type=int, default=0, help="Overlap size (default: 0)")

embed_chunks_parser = subparsers.add_parser("embed_chunks", help="Generate chunk embeddings")

search_chunked_parser = subparsers.add_parser("search_chunked", help="Search using chunk embeddings")
search_chunked_parser.add_argument("query", type=str, help="Search query")
search_chunked_parser.add_argument("--limit", type=int, default=5, help="Number of results to return (default: 5)")


def main():
    
    args = parser.parse_args()
    semantic_instance = SemanticSearch()

    match args.command:
    
        case "verify":
            verify_model()

        case "embed_text":
            embed_text(args.text)

        case "verify_embeddings":
            verify_embeddings()

        case "embedquery":
            embed_query_text(args.embedquery)

        case "search":
            path = os.path.join(os.path.dirname(__file__), "../data/movies.json")
            with open(path, "r") as f:
                movies_data = json.load(f)
            documents = movies_data["movies"]
            semantic_instance.load_or_create_embeddings(documents)
            result = semantic_instance.search(args.query, args.limit)
            for i in range(len(result)):
                print(f"{result[i][1]['title']} (score: {result[i][0]})")

        # just for the test case.
        case "chunk":
            words = args.chunk_text.split()
            chunks = []
            
            if args.overlap and args.overlap > 0:
                i = 0
                while i < len(words):
                    chunk = " ".join(words[i:i + args.chunk_size])
                    chunks.append(chunk)
                    i += args.chunk_size - args.overlap
            else:
                for i in range(0, len(words), args.chunk_size):
                    chunk = " ".join(words[i:i + args.chunk_size])
                    chunks.append(chunk)
            
            print(f"Chunking {len(args.chunk_text)} characters")
            for i, chunk in enumerate(chunks, 1):
                print(f"{i}. {chunk}")


        case"semantic_chunk":

            chunks = semantic_instance.semantic_chunk(args.text, args.max_chunk_size, args.overlap)

            print(f"Semantically chunking {len(args.text)} characters")
            for i, chunk in enumerate(chunks, 1):
                print(f"{i}. {chunk}")

        # it creates embeddings for the movies directly.
        case "embed_chunks":
            
            path = os.path.join(os.path.dirname(__file__), "../data/movies.json")
            with open(path, "r") as f:
                movies_data = json.load(f)
            documents = movies_data["movies"]
            
            chunked_search = ChunkedSemanticSearch()
            embeddings = chunked_search.load_or_create_chunk_embeddings(documents)
            print(f"Generated {len(embeddings)} chunked embeddings")

        # it loads or creates the chunk embeddings,
        # score is printed iteratively.
        case "search_chunked":
            path = os.path.join(os.path.dirname(__file__), "../data/movies.json")
            with open(path, "r") as f:
                movies_data = json.load(f)
            documents = movies_data["movies"]
            
            chunked_search = ChunkedSemanticSearch()
            chunked_search.load_or_create_chunk_embeddings(documents)
            results = chunked_search.search_chunks(args.query, args.limit)
            
            for i, result in enumerate(results, 1):
                print(f"\n{i}. {result['title']} (score: {result['score']:.4f})")
                print(f"   {result['document']}...")

        case _:
            parser.print_help()

if __name__ == "__main__":
    main()