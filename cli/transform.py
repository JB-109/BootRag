from config import stemmer_instance, stop_words_list, table

def transform(query):
    tokens = query.lower().translate(table).split()
    filtered = [word for word in tokens if word not in stop_words_list]
    return [stemmer_instance.stem(word) for word in filtered]