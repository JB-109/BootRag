from config import BM25_B
import pickle
from transform import transform
from config import BM25_K1
from collections import Counter
from collections import defaultdict

import math
import os
import pickle

class InvertedIndex:

    def __init__(self):
        self.index = defaultdict(set)
        self.docmap = {}
        self.term_frequencies = defaultdict(Counter)   
        self.doc_length = {}  

    # inverted idex is built here individually for each doc.
    # this private func is called iteratively for each doc.
    def __add_document(self, doc_id, text):
        token_list = transform(text)
        for token in token_list:
            if not token:
                continue
            self.index[token].add(doc_id)

        self.doc_length[doc_id] = len(token_list)
        self.term_frequencies[doc_id].update(token_list)

    # get a average doc length by dividing the sum of the lengths of all the docs.
    # by the total number of the docs indexed.
    def __get_avg_doc_length(self) -> float:
        if len(self.doc_length) == 0:
            return 0.0
        return sum(self.doc_length.values()) / len(self.doc_length)

#-----------------------------------------------------------------------------
    # this func gets the tf for a given term and a doc.
    def get_tf(self, doc_id, term):
        final_token = transform(term)
        try: 
            if final_token and len(final_token) == 1:
                return self.term_frequencies[doc_id][final_token[0]]
            raise ValueError(f"Term must be a single word, got: '{term}'")

        except Exception as e:
            print(e)

    # b is for length normalization, higher doc length than average gets penalized more.
    # k1 is term frequency satu. lower k1 means diminishing returns for persistent occurances.
    # basically bm25tf applies length norm, and freq satu on the regular tf.
    def get_bm25_tf(self, doc_id, term, k1=BM25_K1, b=BM25_B):
        raw_tf = self.get_tf(doc_id, term)
        length_norm = 1 - b + b * (self.doc_length[doc_id] / self.__get_avg_doc_length())
        return (raw_tf * (k1 + 1)) / (raw_tf + k1 * length_norm)

    # it calculates the score based on how rare the term is among the docs.
    # regular idf produces very high and low scores and senstive to exact document frequencies.
    # whereas bm25idf provides the more conservative score and handles the edge cases of 0 occurances. 
    def get_bm25_idf(self, term) -> float:
        token = transform(term)
        if token and len(token) == 1:
            return math.log((len(self.docmap) - len(self.index[token[0]]) + 0.5) / (len(self.index[token[0]]) + 0.5) + 1)
        raise ValueError(f"Single Token is expected!")
    
    # both are multiplied to get the bm25 score.
    def bm25(self, doc_id, term):
        bm25_tf = self.get_bm25_tf(doc_id, term)
        bm25_idf = self.get_bm25_idf(term)
        return bm25_tf * bm25_idf
    
    # this func tokenizes each term in the query, and runs the bm25 func.
    # then adds to a dict then sorts it in the descending order and return it.
    def bm25_search(self, query, limit=5):
        tokenized_query = transform(query)
        score_dict = defaultdict(float)

        for term in tokenized_query:
            doc_set = self.index[term]
            for doc_id in doc_set:
                score_dict[doc_id] += self.bm25(doc_id, term)

        return dict(sorted(score_dict.items(), key=lambda item: item[1], reverse=True)[:limit])


#-----------------------------------------------------------------------------

    def get_document(self, term):
        return sorted(self.index[term.lower()])
        
    def save(self):
        os.makedirs("cache", exist_ok=True)
        try:
            with open("cache/index.pkl", "wb") as f:
                pickle.dump(self.index, f)
            
            with open("cache/docmap.pkl", "wb") as f:
                pickle.dump(self.docmap, f)

            with open("cache/term_frequencies.pkl", "wb") as f:
                pickle.dump(self.term_frequencies, f)

            with open("cache/doc_lengths.pkl", "wb") as f:
                pickle.dump(self.doc_length ,f)

        except Exception as e:
            print(e)

    def load(self):
        try:

            with open("cache/index.pkl", "rb") as f:
                self.index = pickle.load(f)

            with open("cache/docmap.pkl", "rb") as f:
                self.docmap = pickle.load(f)

            with open("cache/term_frequencies.pkl", "rb") as f:
                self.term_frequencies = pickle.load(f)

            with open("cache/doc_lengths.pkl", "rb") as f:
                self.doc_length = pickle.load(f)

        except Exception as e:
            print(e)

#-----------------------------------------------------------------------------
    # it build the inverted index iteravtively.
    def build(self, movies):
        for each in movies["movies"]:
            self.__add_document(each["id"], f"{each['title']} {each['description']}")
        for each in movies["movies"]:
            self.docmap[each["id"]] = each

