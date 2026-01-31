# RAG Search Engine

A Retrieval-Augmented Generation (RAG) search engine built with Python, aimed at demonstrating the core concepts of modern search and AI-driven answers.

## Features

*   **Hybrid Search**: Combines BM25 (Keyword Search) and Semantic Search (Embeddings) using Reciprocal Rank Fusion (RRF).
*   **RAG Pipeline**: Retrieves relevant movie context and uses Google's Gemini LLM to generate natural language answers.
*   **Evaluation System**: Includes a CLI to measure Precision@K, Recall@K, and F1 scores against a golden dataset.
*   **CLI Interface**: precise and easy to use command line tools.

## Installation

This project uses `uv` for dependency management.

1.  **Clone the repository**:
    ```bash
    git clone <your-repo-url>
    cd rag-search-engine
    ```

2.  **Install dependencies**:
    ```bash
    uv sync
    ```

3.  **Set up Environment Variables**:
    Create a `.env` file in the root directory and add your Google Gemini API Key:
    ```env
    GEMINI_API_KEY=your_api_key_here
    ```

## CLI Reference

All commands are run using `uv run`.

### 1. Augmented Generation (RAG)
The main entry point for the "Chat with Movies" feature.

| Command | Usage | Description |
|---------|-------|-------------|
| **rag** | `uv run cli/augmented_generation_cli.py rag "your query"` | Search and generate an AI answer. |

### 2. Keyword Search (BM25)
Interact with the Inverted Index and BM25 scoring algorithm.

| Command | Usage | Description |
|---------|-------|-------------|
| **search** | `uv run cli/keyword_search_cli.py search "query"` | Basic BM25 search. |
| **build** | `uv run cli/keyword_search_cli.py build` | Rebuild the inverted index. |
| **tf** | `uv run cli/keyword_search_cli.py tf <doc_id> <term>` | Get Term Frequency. |
| **idf** | `uv run cli/keyword_search_cli.py idf <term>` | Get Inverse Document Frequency. |
| **bm25search** | `uv run cli/keyword_search_cli.py bm25search "query"` | Get BM25 scores for a query. |

### 3. Semantic Search (Embeddings)
Interact with the Vector Search and Embeddings.

| Command | Usage | Description |
|---------|-------|-------------|
| **search** | `uv run cli/semantic_search_cli.py search "query"` | Search using whole-document embeddings. |
| **search_chunked** | `uv run cli/semantic_search_cli.py search_chunked "query"` | Search using chunked embeddings (more precise). |
| **embed_text** | `uv run cli/semantic_search_cli.py embed_text "text"` | View the vector for a string. |

### 4. Hybrid Search
Combine Keyword and Semantic search.

| Command | Usage | Description |
|---------|-------|-------------|
| **rrf-search** | `uv run cli/hybrid_search_cli.py rrf-search "query"` | Search using Reciprocal Rank Fusion (Standard). |
| **weighted-search** | `uv run cli/hybrid_search_cli.py weighted-search "query" --alpha 0.5` | Search using Weighted Sum Fusion. |

### 5. Evaluation
Measure the quality of the search engine.

| Command | Usage | Description |
|---------|-------|-------------|
| **Run All** | `uv run cli/evaluation_cli.py` | Run the full test suite against `data/golden_dataset.json`. |

## Project Structure

*   `cli/`: Source code for all search algorithms (BM25, Semantic, Hybrid) and CLI tools.
    *   `augmented_generation_cli.py`: Main entry point for RAG.
    *   `hybrid_search.py`: Implements RRF fusion.
    *   `semantic_search.py`: Handles embeddings and vector search.
    *   `InvertedIndex.py`: Handles keyword matching.
*   `data/`: Contains the `movies.json` dataset and `golden_dataset.json` for testing.

## License

MIT
