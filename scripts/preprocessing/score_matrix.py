from __future__ import division
import pickle
import pandas as pd
import numpy as np

from word_count_evaluation import clean_text, tokenize_and_stem


def map_sets_to_rates(set_name):
    sets_rates = {
        'title_set': 'title_rate',
        'descr_set': 'desc_rate',
        'attr_set': 'attr_rate',
    }
    return sets_rates[set_name]


def create_score_dataframe():
    # # step 1
    # Load bow from previous script
    bow_matrix = pd.read_pickle('../../dataset/bow_per_product.pickle')

    # # Debug
    # print(bow_matrix)

    # # step 2
    # iterate over training data to create the score dataframe
    training_data = pd.read_csv('../../dataset/train.csv', encoding='ISO-8859-1')

    # # debug prints
    # print(bow_matrix)
    # print(training_data)

    score_df = pd.DataFrame(
        columns=['search_id', 'title_rate', 'desc_rate', 'attr_rate', 'relevance'],
        index=training_data['id'].tolist()
    )

    counter = 0
    for isearch in training_data.iterrows():
        search_term_set = set(tokenize_and_stem(clean_text(isearch[1].search_term)))

        # # debug
        # print search_term_set

        # get p_id, search_id and relevance from tr_data
        p_id = isearch[1].product_uid
        relevance = isearch[1].relevance
        search_id = isearch[1].id

        # query the bow_matrix
        sets = {
            'title_set': set(bow_matrix.ix[np.int64(p_id), 'title']),
            'descr_set': set(bow_matrix.ix[np.int64(p_id), 'description']),
            'attr_set': set(bow_matrix.ix[np.int64(p_id), 'attributes']),
        }

        # # debug prints
        # print("")
        # print p_id
        # print relevance

        # print sets['title_set']
        # print sets['descr_set']
        # print sets['attr_set']
        # print search_term_set

        # Instantiate each df row
        score_row = {
            'search_id': search_id,
            'relevance': relevance
        }

        for set_name, iset in sets.iteritems():
            score = calculate_field_score(iset, search_term_set)
            col_name = map_sets_to_rates(set_name)
            score_row[col_name] = score

        score_df.loc[search_id] = pd.Series(score_row)

        if (counter % 1000) == 0:
            print ("Succesfully created " + str(counter) + " rows")
        counter += 1
        # # Debug
        # print(score_df)

    score_df.to_pickle('../../dataset/score_df.pickle')
    print("Score Dataframe succesfully saved!")


def calculate_field_score(field_set, search_set):
    comset = field_set & search_set

    try:
        return len(comset) / len(search_set)
    except ZeroDivisionError:
        print("Division Error occured")
        print("ComSet length: " + str(len(comset)))
        print("SearchSet length: " + str(len(search_set)))
        return len(comset)


if __name__ == "__main__":
    create_score_dataframe()

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
