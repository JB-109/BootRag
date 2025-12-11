import math
from collections import Counter
from collections import defaultdict
import json
import pickle
from nltk.stem.porter import PorterStemmer
import string
import argparse

import sys
import os


def transform(query):
    token_list = query.lower().translate(table).split()
    token_list1 = [stemmer_instance.stem(word) for word in token_list]
    return [word for word in token_list1 if word not in stop_words_list]

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


path = os.path.join(os.path.dirname(__file__), "../data/movies.json")
with open(path, "r") as f:
    # json.load parses the json file and returns a dictionary
    movies_data = json.load(f)

stop_path = os.path.join(os.path.dirname(__file__), "../data/stopwords.txt")
with open(stop_path, "r") as f:
    stop_words_list = f.read().splitlines()

stemmer_instance = PorterStemmer()
table = str.maketrans("", "", string.punctuation)



#-------------------------------------------------------------------

# Here main function is defined as it does not return anything.
# It is used to execute the code.

def main() -> None:
    
    index = InvertedIndex()
    args = parser.parse_args()
    query_result_list = []

    match args.command:
        case "search":

            print(f"Searching for: {args.query}")
            try:
                index.load()
            except Exception as e:
                print(e)
                sys.exit(1)

            query_list = transform(args.query)

        
            for each in query_list:
                if len(query_result_list) < 5: 
                    result = index.get_document(each)
                    if result:
                        query_result_list.extend(result)
            
            final_list = []
            for each in query_result_list[:5]:
                final_list.append(index.docmap[each])

            for i, movie in enumerate(final_list, 1):
                print(f"{i}. {movie['title']}")

        case "build":
            index.build(movies_data)
            index.save()
            print("Index built and saved successfully.")

        case "tf":
            index.load()
            print(index.term_frequencies[args.doc_id][args.term])

        case "idf":
            index.load()
            total_docs = len(index.docmap)
            total_docs_term = len(index.index[stemmer_instance.stem(args.idf_term).lower()])
            idf = math.log((total_docs + 1) / (total_docs_term + 1))
            print(f"Inverse document frequency of '{args.idf_term}': {idf:.2f}")

        case "tfidf":
            index.load()
            total_docs = len(index.docmap)
            total_docs_term = len(index.index[stemmer_instance.stem(args.tfidf_term).lower()])
            idf = math.log((total_docs + 1) / (total_docs_term + 1))
            tf = index.term_frequencies[args.tfidf_doc_id][stemmer_instance.stem(args.tfidf_term).lower()]
            tf_idf = idf * tf
            print(f"TF-IDF score of '{args.tfidf_term}' in document '{args.tfidf_doc_id}': {tf_idf:.2f}")

        case _:
            parser.print_help()




#----------------------------------------------------------------




class InvertedIndex:
    def __init__(self):
        self.index = defaultdict(set)
        self.docmap = {}
        self.term_frequencies = defaultdict(Counter)     

    def __add_document(self, doc_id, text):
        token_list = text.lower().translate(table).split()
        token_list1 = [stemmer_instance.stem(word) for word in token_list]
        token_list2 = [word for word in token_list1 if word not in stop_words_list]
        
        for token in token_list2:
            if not token:
                continue
            self.index[token].add(doc_id)

        self.term_frequencies[doc_id].update(token_list2)

    def get_document(self, term):
        return sorted(self.index[term.lower()])

    def build(self, movies):
        for each in movies["movies"]:
            self.__add_document(each["id"], f"{each['title']} {each['description']}")
        for each in movies["movies"]:
            self.docmap[each["id"]] = each

    def save(self):
        os.makedirs("cache", exist_ok=True)
        
        with open("cache/index.pkl", "wb") as f:
            pickle.dump(self.index, f)
        
        with open("cache/docmap.pkl", "wb") as f:
            pickle.dump(self.docmap, f)

        with open("cache/term_frequencies.pkl", "wb") as f:
            pickle.dump(self.term_frequencies, f)

    def load(self):
        try:

            with open("cache/index.pkl", "rb") as f:
                self.index = pickle.load(f)
            with open("cache/docmap.pkl", "rb") as f:
                self.docmap = pickle.load(f)
            with open("cache/term_frequencies.pkl", "rb") as f:
                self.term_frequencies = pickle.load(f)

        except Exception as e:
            print(e)
    
    def get_tf(self, doc_id, term):
        final_token = term.lower().translate(table).split()

        try: 
            if final_token and len(final_token) == 1:
                return self.term_frequencies[doc_id][stemmer_instance.stem(final_token[0])]
            raise ValueError(f"Term must be a single word, got: '{term}'")

        except Exception as e:
            print(e)

#-------------------------------------------------------------------

if __name__ == "__main__":
    main()