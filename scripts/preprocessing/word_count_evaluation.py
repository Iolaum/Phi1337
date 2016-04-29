import pandas as pd
import re
import nltk
import pickle

from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer


def clean_text(text):
    clean_text = re.sub("[^a-zA-Z0-9]]", " ", text, 0, re.UNICODE)
    words = clean_text.lower().split()

    stops = set(nltk.corpus.stopwords.words("english"))

    meaningful_words = [w for w in words if w not in stops]

return " ".join(meaningful_words)


def tokenize_and_stem(text):
stemmer = SnowballStemmer("english")

# first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]
filtered_tokens = []

# filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
for token in tokens:
if re.search('[a-zA-Z]', token):
    filtered_tokens.append(token)
stems = [stemmer.stem(t) for t in filtered_tokens]

return " ".join(stems)


def main():
"Read Files"
default_atrr_name = "bullet"
# get the title from the matrix
training_data = pd.read_csv("../../dataset/train.csv", encoding="ISO-8859-1")
descriptions = pd.read_csv("../../dataset/product_descriptions.csv")
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

    for prod_id in prod_ids:
        product_title = training_data.loc[training_data['product_uid'] == prod_id].iloc[0]['product_title']
        product_description = descriptions.loc[descriptions['product_uid'] == prod_id].iloc[0]['product_description']
        prod_attributes = attributes.loc[attributes['product_uid'] == prod_id]

        # print(tokenize_and_stem(clean_text(product_title)))
        # print(product_description)
        # print(prod_attributes.shape)
        attrs = []
        for i,r in prod_attributes.iterrows():
            if r['name'].lower().find(default_atrr_name) != -1:
                attrs.append(r['value'])
            else:
                attrs.append(' '.join([r['name'], r['value']]))
        all_attributes = ' '.join(attrs)
        
        bag_of_word_matrix[bow_col0].append(prod_id)
        bag_of_word_matrix[bow_col1].append(tokenize_and_stem(clean_text(product_title)))
        bag_of_word_matrix[bow_col2].append(tokenize_and_stem(clean_text(product_description)))
        bag_of_word_matrix[bow_col3].append(tokenize_and_stem(clean_text(all_attributes)))
        
        print bag_of_word_matrix
        break
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
    main()
