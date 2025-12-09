import json
import argparse
import os
import string

from nltk.stem import PorterStemmer

path = os.path.join(os.path.dirname(__file__), "../data/movies.json")
with open(path, "r") as f:
    data = json.load(f)

stop_path = os.path.join(os.path.dirname(__file__), "../data/stopwords.txt")
with open(stop_path, "r") as f:
    stop_data = f.read().splitlines()

stemmer = PorterStemmer()

def main() -> None:
    parser = argparse.ArgumentParser(description="Keyword Search CLI")
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    search_parser = subparsers.add_parser("search", help="Search movies using BM25")
    search_parser.add_argument("query", type=str, help="Search query")

    args = parser.parse_args()
    result_list = []

    match args.command:
        case "search":

            print(f"Searching for: {args.query}")

            table = str.maketrans("", "", string.punctuation)

            for each in data["movies"]:
                if len(result_list) < 5:
                    
                    args_list = args.query.lower().translate(table).split()
                    title_list = each["title"].lower().translate(table).split()

                    args_list = [stemmer.stem(word) for word in args_list]
                    title_list = [stemmer.stem(word) for word in title_list]
                        
                    args_list = [word for word in args_list if word not in stop_data]
                    title_list = [word for word in title_list if word not in stop_data]

                    match_found = False
                    for ar in args_list:
                        for ti in title_list:
                            if ar in ti:
                                match_found = True
                                result_list.append(each)
                                break
                        if match_found:
                            break

            for i in range(1, len(result_list)+1):
                print(f"{i}. {result_list[i-1]["title"]}")

        case _:
            parser.print_help()


if __name__ == "__main__":
    main()