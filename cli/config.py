import argparse
from nltk.stem.porter import PorterStemmer

import os
import json
import string

# Here 2nd Argument is the path relative to the current file,
# No matter from where the curren file is executed, this will always point to the correct path.

BM25_K1 = 1.5
BM25_B = 0.75

parser = argparse.ArgumentParser(description="Keyword Search CLI")
subparsers = parser.add_subparsers(dest="command", help="Available commands")

search_parser = subparsers.add_parser("search", help="Search movies using BM25")
search_parser.add_argument("query", type=str, help="Search query")

build_parser = subparsers.add_parser("build", help="Build and save the inverted index")

term_parser = subparsers.add_parser("tf", help="Gives the term frequency")
term_parser.add_argument("doc_id", type=int, help="document ID")
term_parser.add_argument("term", type=str, help="Literal term")

idf_parser = subparsers.add_parser("idf", help="Inverse Document Frequency")
idf_parser.add_argument("idf_term", type=str, help="Literal Term")

tfidf_parser = subparsers.add_parser("tfidf", help="Gives tfidf")
tfidf_parser.add_argument("tfidf_doc_id", type=int, help="document ID")
tfidf_parser.add_argument("tfidf_term", type=str, help="Literal term")

bm25idf_parser = subparsers.add_parser("bm25idf", help="bm25")
bm25idf_parser.add_argument("bm25_term", type=str, help="Literal Term")

bm25tf_parser = subparsers.add_parser("bm25tf", help="bm25")
bm25tf_parser.add_argument("bm25tf_doc_id", type=int, help="help me")
bm25tf_parser.add_argument("bm25tf_term", type=str, help="Literal Term")
bm25tf_parser.add_argument("k1", type=float, nargs='?', default=BM25_K1, help="Tunable BM25 K1 parameter")
bm25tf_parser.add_argument("b", type=float, nargs='?', default=BM25_B, help="Tunable BM25 b parameter")

bm25_parser = subparsers.add_parser("bm25search", help="bm25 score")
bm25_parser.add_argument("bm25_query", type=str, help="Actual Query")
bm25_parser.add_argument("bm25_limit", type=int, nargs="?", default=5, help="limited result")

path = os.path.join(os.path.dirname(__file__), "../data/movies.json")
with open(path, "r") as f:
    # json.load parses the json file and returns a dictionary
    movies_data = json.load(f)

stop_path = os.path.join(os.path.dirname(__file__), "../data/stopwords.txt")
with open(stop_path, "r") as f:
    stop_words_list = f.read().splitlines()

stemmer_instance = PorterStemmer()
table = str.maketrans("", "", string.punctuation)