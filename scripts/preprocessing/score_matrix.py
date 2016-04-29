import pickle
import pandas as pd
import numpy as np


def create_score_matrix():
    # with open('../../dataset/bow_matrix.p', 'rb') as handle:
    # bow_matrix = pickle.load(handle)
    # hard coded for testing
    bow_matrix = {
        '100001': {
            'product_title': [u'simpson', u'strong-ti', u'12-gaug', u'angl'],
            'product_description': [u'angl', u'make', u'joint', u'stronger', u'also', u'provid', u'consist',
                                    u'straight', u'corner', u'simpson', u'strong-ti', u'offer', u'wide', u'varieti',
                                    u'angl', u'various', u'size', u'thick', u'handl', u'light-duti', u'job', u'project',
                                    u'structur', u'connect', u'need', u'bent', u'skew', u'match', u'project',
                                    u'outdoor', u'project', u'moistur', u'present', u'use', u'zmax', u'zinc-coat',
                                    u'connector', u'provid', u'extra', u'resist', u'corros', u'look', 'z', u'end',
                                    u'model', u'number', u'.versatil', u'connector', u'various', u'connect', u'home',
                                    u'repair', u'projectsstrong', u'angl', u'nail', u'screw', u'fasten', u'alonehelp',
                                    u'ensur', u'joint', u'consist', u'straight', u'strongdimens', 'in', 'x', 'in', 'x',
                                    u'in.mad', u'12-gaug', u'steelgalvan', u'extra', u'corros', u'resistanceinstal',
                                    u'10d', u'common', u'nail', 'x', 'in', u'strong-driv', 'sd', u'screw'],
            'product_attributes': []
        },
        '100002': {
            'product_title': [u'behr', u'premium', u'textur', u'deckov', u'1-gal', u'sc-141', u'tugboat', u'wood',
                              u'concret', u'coat'],
            'product_description': [u'behr', u'premium', u'textur', u'deckov', u'innov', u'solid', u'color', u'coat',
                                    u'bring', u'old', u'weather', u'wood', u'concret', u'back', u'life', u'advanc',
                                    u'acryl', u'resin', u'formula', u'creat', u'durabl', u'coat', u'tire', u'worn',
                                    u'deck', u'rejuven', u'whole', u'new', u'look', u'best', u'result', u'sure',
                                    u'proper', u'prepar', u'surfac', u'use', u'applic', u'behr', u'product', u'display',
                                    u'above.california', u'resid', u'see', u'nbsp', u'proposit', u'informationrev',
                                    u'wood', u'composit', u'deck', u'rail', u'porch', u'boat', u'dock', u'also',
                                    u'great', u'concret', u'pool', u'deck', u'patio', u'sidewalks100', u'acryl',
                                    u'solid', u'color', u'coatingresist', u'crack', u'peel', u'conceal', u'splinter',
                                    u'crack', u'in.provid', u'durabl', u'mildew', u'resist', u'finishcov', 'sq', u'ft.',
                                    u'coat', u'per', u'galloncr', u'textur', u'slip-resist', u'finishfor', u'best',
                                    u'result', u'prepar', u'appropri', u'behr', u'product', u'wood', u'concret',
                                    u'surfaceactu', u'paint', u'color', u'may', u'vari', u'on-screen', u'printer',
                                    u'representationscolor', u'avail', u'tint', u'storesonlin', u'price', u'includ',
                                    u'paint', u'care', u'fee', u'follow', u'state', 'ca', 'co', 'ct', 'me', 'mn', 'or',
                                    'ri', 'vt'],
            'product_attributes': []
        },
        '100005': {
            'product_title': [u'delta', u'vero', u'1-handl', u'shower', u'faucet', u'trim', u'kit', u'chrome', u'valv',
                              u'includ'],
            'product_description': [u'updat', u'bathroom', u'delta', u'vero', u'single-handl', u'shower', u'faucet',
                                    u'trim', u'kit', u'chrome', u'sleek', u'modern', u'minimalist', u'aesthet',
                                    u'multichoic', u'univers', u'valv', u'keep', u'water', u'temperatur', u'within',
                                    u'degre', u'fahrenheit', u'help', u'prevent', u'scalding.california', u'resid',
                                    u'see', u'nbsp', u'proposit', u'informationinclud', u'trim', u'kit', u'onli',
                                    u'rough-in', u'kit', u'r10000-unbx', u'sold', u'separatelyinclud',
                                    u'handlemaintain', u'balanc', u'pressur', u'hot', u'cold', u'water', u'even',
                                    u'valv', u'turn', u'elsewher', u'systemdu', u'watersens', u'regul', u'state',
                                    u'new', u'york', u'pleas', u'confirm', u'ship', u'zip', u'code', u'restrict',
                                    u'use', u'item', u'meet', u'watersens', u'qualif'],
            'product_attributes': []
        },
    }

    score_matrix = np.ndarray()

    training_data = pd.read_csv("../../dataset/train.csv", encoding="ISO-8859-1")
    print(bow_matrix)
    print(training_data)

    for search in training_data:
        pass
        # search_id = ....
        # search_term = ....
        # search_relevance = ....
        # title_score = calculate_field_score(bow_matrix[product_title], search_term)
        # attr_score = calculate_field_score(bow_matrix[product_attributes], search_term)
        # desc_score = calculate_field_score(bow_matrix[product_description], search_term)

        # Insert the above into our score_matrix


def calculate_field_score(bow_list_one, bow_list_two):
    pass


if __name__ == "__main__":
    create_score_matrix()

#
# def perform_tf_idf():
#     print("Getting cleaned books...")
#
#     max_features = 50000
#
#     # define vectorizer parameters
#     print("Setup TF-IDF Vectorizer")
#     tfidf_vectorizer = TfidfVectorizer(max_df=0.7, max_features=None,
#                                        min_df=0.2, stop_words=None,
#                                        use_idf=True, tokenizer=None)
#
#     print("Perform TF-IDF on the books -- Max features = " + str(max_features))
#
#     tfidf_matrix = tfidf_vectorizer.fit_transform(books)  # fit the vectorizer to books
#     print(tfidf_matrix.shape)
#
#     return tfidf_matrix, tfidf_vectorizer
