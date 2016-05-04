import pandas as pd
import numpy as np
import os

from a02a_word_count_evaluation import clean_text
from a01c_feature_engineering import tokenize_and_stem

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import linear_kernel
from sklearn.metrics.pairwise import polynomial_kernel
from sklearn.metrics.pairwise import sigmoid_kernel
from sklearn.metrics.pairwise import rbf_kernel
from sklearn.metrics.pairwise import laplacian_kernel

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import TruncatedSVD


def get_kernel_types():
    return [
        'cosine_similarity',
        'linear',
        'polynomial',
        'sigmoid',
        'rbf',
        'laplacian',
    ]


def get_terms_from_tf_idf(tfidf_vectorizer):
    terms = tfidf_vectorizer.get_feature_names()

    return terms


def get_kernel(kernel_name):
    options = {
        'cosine_similarity': cosine_similarity,
        'linear': linear_kernel,
        'polynomial': polynomial_kernel,
        'sigmoid': sigmoid_kernel,
        'rbf': rbf_kernel,
        'laplacian': laplacian_kernel
    }

    return options[kernel_name]


def get_similarity_matrix(tfidf_matr, kernel):
    dist = 1 - get_kernel(kernel)(tfidf_matr)

    return dist


def perform_lsa_count_vect(debug):
    cvect = CountVectorizer(min_df=0)

    doc_matrix = pd.read_pickle('../../dataset/bow_per_product.pickle')
    # if debug:
    #     print doc_matrix

    training_data = pd.read_csv('../../dataset/preprocessed_training_data.csv')

    all_feature_names = [0, 1, 2, 3, 'relevance']
    score_df = pd.DataFrame(
        columns=all_feature_names,
        index=training_data['id'].tolist()
    )

    print("Starting Count Vectorizer!")

    counter = 0
    for isearch in training_data.iterrows():

        search_term_tokens = isearch[1].search_term

        # # debug
        # print search_term_set

        # get p_id, search_id and relevance from tr_data
        p_id = isearch[1].product_uid
        relevance = isearch[1].relevance
        search_id = isearch[1].id

        test_matrix = [
            search_term_tokens,
            " ".join(doc_matrix.ix[np.int64(p_id), 'title']),
            " ".join(doc_matrix.ix[np.int64(p_id), 'description']),
            " ".join(doc_matrix.ix[np.int64(p_id), 'attributes']),
        ]

        # count vectorizer to books
        cvect_matrix = cvect.fit_transform(test_matrix)

        # # debug
        # print cvect_matrix
        # print type(cvect_matrix)
        # print cvect_matrix.shape

        lsa = TruncatedSVD(n_components=10, random_state=9)

        lsa_matrix = lsa.fit_transform(cvect_matrix)

        # # debug
        # print lsa_matrix
        # print type(lsa_matrix)

        # # Debug
        # print(sim_matrix)
        # print("Title score - " + str(title_score))
        # print("Desc score - " + str(desc_score))
        # print("Attrs score - " + str(attr_score))

        score_row = {
            'relevance': relevance,
        }
        # debug


        for key, value in enumerate(lsa_matrix[0]):
            score_row[key] = value

            #     if debug:
            #         print key
            #         print value

            # # Debug
            # print("Score row {} is:\n{}".format(counter,score_row))

            # if counter == 2:
        #     break

        score_df.loc[search_id] = pd.Series(score_row)
        # print("pd.series score_row is:\n{}".format(pd.Series(score_row)))

        counter += 1

        if counter is not 0 and counter % 1000 == 0:
            print(str(counter) + " searches processed")
            # # Stop execution for debug reasons
            # if counter == 1000:
            #     break

    score_df.to_pickle('../../dataset/score_df_lsa_cvect.pickle')

    if debug:
        print(score_df)

    print("Score Dataframe lsa_cvect succesfully saved!")
    return None

if __name__ == "__main__":
    perform_lsa_count_vect(debug=True)


# Calculate similarity with - cosine_similarity kernel
# [[  0.00000000e+00   1.00000000e+00   1.00000000e+00   1.00000000e+00]

# Calculate similarity with - linear kernel
# [[  0.00000000e+00   1.00000000e+00   1.00000000e+00   1.00000000e+00]

# Calculate similarity with - polynomial kernel
# [[-0.03946922  0.          0.          0.        ]

# Calculate similarity with - sigmoid kernel
# [[ 0.23300535  0.23840584  0.23840584  0.23840584]

# Calculate similarity with - rbf kernel
# [[ 0.          0.01290305  0.0256396   0.0256396 ]

# Calculate similarity with - laplacian kernel
# [[ 0.          0.01290305  0.10028499  0.07684025]
