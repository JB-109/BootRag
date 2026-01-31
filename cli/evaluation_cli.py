import argparse
import json
import os
from hybrid_search import HybridSearch


def main():
    parser = argparse.ArgumentParser(description="Search Evaluation CLI")
    parser.add_argument("--limit", type=int, default=5, help="Number of results to evaluate (k for precision@k, recall@k)")

    args = parser.parse_args()
    limit = args.limit

    golden_path = os.path.join(os.path.dirname(__file__), "../data/golden_dataset.json")
    with open(golden_path, "r") as f:
        golden_data = json.load(f)
    
    movies_path = os.path.join(os.path.dirname(__file__), "../data/movies.json")
    with open(movies_path, "r") as f:
        movies_data = json.load(f)["movies"]
    
    hybrid_search = HybridSearch(movies_data)
    
    # Process each test case
    for test_case in golden_data["test_cases"]:
        query = test_case["query"]
        relevant_docs = test_case["relevant_docs"]
        
        results = hybrid_search.rrf_search(query, k=60, limit=limit)
        
        retrieved_titles = [result["title"] for result in results]
        
        # Calculate precision
        # Precision = number of relevant docs in retrieved / total retrieved
        # rrf returns only the limit movies, unless rerank is mentioned by the user.
        relevant_found = [title for title in retrieved_titles if title in relevant_docs]
        precision = len(relevant_found) / limit if limit > 0 else 0.0
        recall = len(relevant_found) / len(relevant_docs)
        f1 = 2 * (precision * recall) / (precision + recall)

        # Print metric
        print(f"- Query: {query}")
        print(f"  - Precision@{limit}: {precision:.4f}")
        print(f"  - Recall@{limit}: {recall:.4f}")
        print(f"  - F1 Score: {f1:.4f}")
        print(f"  - Retrieved: {', '.join(retrieved_titles)}")
        print(f"  - Relevant Found: {', '.join(relevant_found)}")
        print(f"  - Relevant: {', '.join(relevant_docs)}")
        print()



if __name__ == "__main__":
    main()