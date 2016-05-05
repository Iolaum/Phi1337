import pandas as pd
import re
import nltk
import pickle
import os

from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from a01c_feature_engineering import preprocess_data

stops = set(nltk.corpus.stopwords.words("english"))

def clean_text(text):
    clean_text = re.sub("[^a-zA-Z0-9]]", " ", re.sub(r'[^\x00-\x7f]', r'', text), 0, re.UNICODE)
    words = clean_text.lower().split()

    meaningful_words = [w for w in words if w not in stops]

    return " ".join(meaningful_words)

def main(return_text=False):
    # Read Files
    default_atrr_name = "bullet"
    # get the title from the matrix
    training_data = preprocess_data()
    # columns name : [u'product_uid', u'name', u'value']
    attributes = pd.read_csv("../../dataset/attributes.csv")

    # column names : 'title', 'description', 'attributes'
    bow_col0 = 'product_uid'
    bow_col1 = 'title'
    bow_col2 = 'description'
    bow_col3 = 'attributes'
    column_orders = [bow_col0, bow_col1, bow_col2, bow_col3]
    bag_of_word_matrix = dict({bow_col0: [], bow_col1: [], bow_col2: [], bow_col3: []})
    prod_ids = training_data["product_uid"].unique()
    counter = 0
    max_count = -1

    for prod_id in prod_ids:
        product_title = training_data.loc[training_data['product_uid'] == prod_id].iloc[0]['product_title']
        product_description = training_data.loc[training_data['product_uid'] == prod_id].iloc[0]['product_description']
        prod_attributes = attributes.loc[attributes['product_uid'] == prod_id]

        # print(tokenize_and_stem(clean_text(product_title)))
        # print(product_description)
        # print(prod_attributes.shape)
        attrs = []
        for i, r in prod_attributes.iterrows():
            if r['name'].lower().find(default_atrr_name) != -1:
                attrs.append(r['value'])
            else:
                str1 = str(r['name'])
                str2 = str(r['value'])
                mixed_str = []
                if len(str1) > 0:
                    mixed_str.append(str1)
                if len(str2) > 0:
                    mixed_str.append(str2)

                attrs.append(' '.join(mixed_str))
        all_attributes = ' '.join(attrs)

        bag_of_word_matrix[bow_col0].append(prod_id)
        bag_of_word_matrix[bow_col1].append(product_title.split())
        bag_of_word_matrix[bow_col2].append(product_description.split())
        bag_of_word_matrix[bow_col3].append(all_attributes.split())

        counter += 1
        if counter == max_count:
            break
        if counter % 1000 == 0:
            print("Processed " + str(counter) + " entries.")

    # create panda dataframe
    df = pd.DataFrame(bag_of_word_matrix, index=prod_ids.tolist()[:counter], columns=column_orders)

    # print type(df.index.values[0])
    # print type(df.index[0])
    df.to_pickle('../../dataset/bow_per_product.pickle')
    print(" Finished creating bag of word matrix!")

    # for prod_attr in prod_attributes:
    #     print(prod_attr)

    # testing_data = pd.read_csv("test.csv", encoding="ISO-8859-1")

    # for desc in descriptions:
    #     print(desc)
    #     clean_description = clean_text(desc)
    #     stemmed_desc = tokenize_and_stem(clean_description)
    #     print(stemmed_desc)
    #     exit()

    # attribute_data = pd.read_csv("attributes.csv")


if __name__ == "__main__":
    # Change return_text to decide if the cleaned result of each text will be text or list
    if os.path.isfile("../../dataset/bow_per_product.pickle"):
        print("Found Bag of Words DataFrame... No Need to proceed further.")
    else:
        print("No Bag of Words DataFrame Found... Proceed BoW creation")
        main(return_text=False)
