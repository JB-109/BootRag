from config import stemmer_instance, stop_words_list, table

def transform(query):
    # 1. lowercase + remove punctuation
    tokens = query.lower().translate(table).split()

    # 2. remove stopwords first
    filtered = [word for word in tokens if word not in stop_words_list]

    # 3. then stem
    return [stemmer_instance.stem(word) for word in filtered]