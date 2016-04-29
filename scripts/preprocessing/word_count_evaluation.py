import pandas as pd
import re
import nltk
from nltk.stem.snowball import SnowballStemmer


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
    # get the title from the matrix
    training_data = pd.read_csv("../../dataset/train.csv", encoding="ISO-8859-1")
    descriptions = pd.read_csv("../../dataset/product_descriptions.csv")
    attributes = pd.read_csv("../../dataset/product_descriptions.csv")

    bag_of_word_matrix = dict()
    prod_ids = training_data["product_uid"]
    for prod_id in prod_ids:
        # product_title = training_data.loc[training_data['product_uid'] == prod_id].iloc[0]['product_title']
        # product_description = descriptions.loc[descriptions['product_uid'] == prod_id].iloc[0]['product_description']
        prod_attributes = attributes.loc[attributes['product_uid'] == prod_id]

        print(prod_attributes)
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
