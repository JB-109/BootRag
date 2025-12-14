from semantic_search import SemanticSearch
from semantic_search import embed_query_text
from semantic_search import embed_text
from semantic_search import verify_model
from semantic_search import verify_embeddings
import argparse
import os
import json

def main():
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

        case _:
            parser.print_help()

if __name__ == "__main__":
    main()