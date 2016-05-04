import pandas as pd
import numpy as np
import os

from word_count_evaluation import clean_text
from feature_engineering import tokenize_and_stem

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

    doc_matrix = pd.read_csv('../../dataset/bow_per_product.csv')
    # if debug:
    #     print doc_matrix

    training_data = pd.read_csv('../../dataset/train.csv', encoding='ISO-8859-1')

    all_feature_names = [0, 1, 2, 3, 'relevance']
    score_df = pd.DataFrame(
        columns=all_feature_names,
        index=training_data['id'].tolist()
    )

    print("Starting Count Vectorizer!")

    counter = 0
    for isearch in training_data.iterrows():

        search_term_tokens = tokenize_and_stem(clean_text(isearch[1].search_term), return_text=True)

        # # debug
        # print search_term_set

        # get p_id, search_id and relevance from tr_data
        p_id = isearch[1].product_uid
        relevance = isearch[1].relevance
        search_id = isearch[1].id

        test_matrix = [
            search_term_tokens,
            (" ".join(doc_matrix.ix[np.int64(p_id), 'title'])).decode('ISO-8859-1'),
            (" ".join(doc_matrix.ix[np.int64(p_id), 'description'])).decode('ISO-8859-1'),
            (" ".join(doc_matrix.ix[np.int64(p_id), 'attributes'])).decode('ISO-8859-1'),
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

    score_df.to_csv('../../dataset/score_df_lsa_cvect.csv')

    if debug:
        print(score_df)

    print("Score Dataframe lsa_cvect succesfully saved!")
    return None


def perform_tf_idf(debug=False):
    bow_matrix = pd.read_csv('../../dataset/bow_per_product.csv')

    max_features = None
    # define vectorizer parameters
    print("Setup TF-IDF Vectorizer")

    tfidf_vectorizer = TfidfVectorizer(max_df=0.7, max_features=max_features,
                                       min_df=0.2, stop_words=None,
                                       use_idf=True, tokenizer=None)

    print("Perform TF-IDF on the search results -- Max features = " + str(max_features))
    kernel_type = 'rbf'

    training_data = pd.read_csv('../../dataset/train.csv', encoding='ISO-8859-1')

    # # debug prints
    #     print("Bag of words matrix: ")
    #     print(bow_matrix)
    #     print("")
    #     print("Training Data: ")
    #     print(training_data)

    score_df = pd.DataFrame(
        columns=['title_rate', 'desc_rate', 'attr_rate', 'relevance'],
        index=training_data['id'].tolist()
    )

    counter = 0
    for isearch in training_data.iterrows():
        search_term_tokens = tokenize_and_stem(clean_text(isearch[1].search_term))

        # # debug
        # print search_term_set

        # get p_id, search_id and relevance from tr_data
        p_id = isearch[1].product_uid
        relevance = isearch[1].relevance
        search_id = isearch[1].id

        test_matrix = [
            " ".join(search_term_tokens),
            " ".join(bow_matrix.ix[np.int64(p_id), 'title']),
            " ".join(bow_matrix.ix[np.int64(p_id), 'description']),
            " ".join(bow_matrix.ix[np.int64(p_id), 'attributes']),
        ]

        tfidf_matrix = tfidf_vectorizer.fit_transform(test_matrix)  # fit the vectorizer to books

        # Run all kernels for debug reasons (see print below)
        #
        # for kernel_type in get_kernel_types():
        #     print("Calculate similarity with - " + kernel_type + " kernel")
        #     sim_matrix = get_similarity_matrix(tfidf_matrix, kernel_type)[0]
        #     print(sim_matrix)
        # break

        # # Debug
        # print("Calculate similarity with - " + kernel_type + " kernel")

        sim_matrix = get_similarity_matrix(tfidf_matrix, kernel_type)[0]
        title_score = sim_matrix[1]
        desc_score = sim_matrix[2]
        attr_score = sim_matrix[3]

        # # Debug
        # print(sim_matrix)
        # print("Title score - " + str(title_score))
        # print("Desc score - " + str(desc_score))
        # print("Attrs score - " + str(attr_score))

        score_row = {
            'relevance': relevance,
            'title_rate': title_score,
            'desc_rate': desc_score,
            'attr_rate': attr_score,
        }

        score_df.loc[search_id] = pd.Series(score_row)

        counter += 1

        if (counter is not 0 and counter % 1000 == 0):
            print(str(counter) + " searches processed")
            # # Stop execution for debug reasons
            # if counter == 1000:
            #     break

    score_df.to_csv('../../dataset/score_df_tfidf.csv')

    if debug:
        print(score_df)

    print("Score Dataframe succesfully saved!")
    return None


if __name__ == "__main__":
    # perform_tf_idf(debug=True)
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
