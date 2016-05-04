from __future__ import division
import pickle
import pandas as pd
import numpy as np
import os

from word_count_evaluation import clean_text
from feature_engineering import tokenize_and_stem


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
    bow_matrix = pd.read_csv('../../dataset/bow_per_product.csv')

    # # Debug
    # print(bow_matrix)

    # # step 2
    # iterate over training data to create the score dataframe
    training_data = pd.read_csv('../../dataset/train.csv', encoding='ISO-8859-1')

    # # debug prints
    # print(bow_matrix)
    # print(training_data)

    all_feature_names = ['title_rate', 'desc_rate', 'attr_rate', 'relevance']
    score_df = pd.DataFrame(
        columns=all_feature_names,
        index=training_data['id'].tolist()
    )

    counter = 0
    for isearch in training_data.iterrows():
        search_term_set = set(isearch[1].search_term.split())

        # # debug
        # print search_term_set

        # get p_id, search_id and relevance from tr_data
        p_id = isearch[1].product_uid
        np_pid = np.int64(p_id)
        relevance = isearch[1].relevance
        search_id = isearch[1].id

        # query the bow_matrix
        sets = {
            'title_set': set(bow_matrix.ix[np_pid, 'title']),
            'descr_set': set(bow_matrix.ix[np_pid, 'description']),
            'attr_set': set(bow_matrix.ix[np_pid, 'attributes']),
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

    print score_df.shape

    score_df.to_csv('../../dataset/score_df.csv')
    print("Score Dataframe succesfully saved!")


def calculate_field_score(field_set, search_set):
    comset = field_set & search_set

    try:
        return len(comset) / len(search_set)
    except ZeroDivisionError:
        print("Division Error occured")
        return len(comset)


if __name__ == "__main__":
    create_score_dataframe()
