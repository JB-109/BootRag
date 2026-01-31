import argparse
import os
import json
from dotenv import load_dotenv
from google import genai
from hybrid_search import HybridSearch


parser = argparse.ArgumentParser(description="Retrieval Augmented Generation CLI")
subparsers = parser.add_subparsers(dest="command", help="Available commands")

rag_parser = subparsers.add_parser("rag", help="Perform RAG (search + generate answer)")
rag_parser.add_argument("query", type=str, help="Search query for RAG")

args = parser.parse_args()

def main():
    
    match args.command:
        case "rag":
            query = args.query
            
            # Load movies data
            path = os.path.join(os.path.dirname(__file__), "../data/movies.json")
            with open(path, "r") as f:
                movies_data = json.load(f)
            documents = movies_data["movies"]
            
            # Perform RRF search
            hybrid_search = HybridSearch(documents)
            results = hybrid_search.rrf_search(query, k=60, limit=5)
            
            # Format documents for the prompt
            docs = ""
            for i, result in enumerate(results, 1):
                docs += f"{i}. {result['title']}\n"
                docs += f"   {result.get('document', '')}\n\n"
            
            # Generate answer using LLM
            load_dotenv()
            api_key = os.environ.get("GEMINI_API_KEY")
            client = genai.Client(api_key=api_key)
            
            prompt = f"""Answer the question or provide information based on the provided documents. This should be tailored to Hoopla users. Hoopla is a movie streaming service.
                        Query: {query}
                        Documents:
                        {docs}
                        Provide a comprehensive answer that addresses the query:"""
            
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt
            )
            
            # Print results
            print("Search Results:")
            for result in results:
                print(f"  - {result['title']}")
            
            print("\nRAG Response:")
            print(response.text.strip())
        


        case _:
            parser.print_help()


if __name__ == "__main__":
    main()
