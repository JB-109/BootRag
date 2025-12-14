from config import stemmer_instance
from config import movies_data
from transform import transform
from config import parser
from InvertedIndex import InvertedIndex
import math
import sys


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
            print(index.get_tf(args.doc_id, args.term))

        case "idf":
            index.load()
            total_docs = len(index.docmap)
            total_docs_term = len(index.index[transform(args.idf_term)[0]])

            idf = math.log((total_docs + 1) / (total_docs_term + 1))

            print(f"Inverse document frequency of '{args.idf_term}': {idf:.2f}")

        case "tfidf":
            index.load()
            total_docs = len(index.docmap)
            total_docs_term = len(index.index[transform(args.tfidf_term)[0]])
            
            idf = math.log((total_docs + 1) / (total_docs_term + 1))
            tf = index.get_tf(args.tfidf_doc_id, args.tfidf_term)
            tf_idf = idf * tf

            print(f"TF-IDF score of '{args.tfidf_term}' in document '{args.tfidf_doc_id}': {tf_idf:.2f}")

        case "bm25idf":
            index.load()
            call_result = index.get_bm25_idf(args.bm25_term)
            print(f"BM25 IDF score of '{args.bm25_term}': {call_result:.2f}")

        case "bm25tf":
            index.load()
            result = index.get_bm25_tf(args.bm25tf_doc_id, args.bm25tf_term, args.k1, args.b)
            print(f"BM25 TF score of '{args.bm25tf_term}' in document '{args.bm25tf_doc_id}': {result:.2f}")

        case "bm25search":
            index.load()
            result = index.bm25_search(args.bm25_query, args.bm25_limit)
            for item in result.items():
                print(f"({item[0]}) {index.docmap[item[0]]['title']} - Score: {item[1]:.2f}")

        case _:
            parser.print_help()


#-------------------------------------------------------------------

if __name__ == "__main__":
    main()