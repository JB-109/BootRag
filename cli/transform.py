from config import stemmer_instance, stop_words_list, table

def transform(query):
    token_list = query.lower().translate(table).split()
    token_list1 = [stemmer_instance.stem(word) for word in token_list]
    return [word for word in token_list1 if word not in stop_words_list]