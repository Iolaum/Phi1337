# -*- coding: utf-8 -*-

import numpy as np
import os
import pandas as pd
import re
import nltk
import pickle
from nltk.stem.porter import *
from nltk.stem.snowball import SnowballStemmer

from unidecode import unidecode

def fixtypos(training_data):
    # traing_data to be given when called

    with open("../../dataset/misstypo.p", 'rb') as f:
        dic = pickle.load(f)

    print("Started replacing typos in search terms")
    print("This may take a while...")
    training_data['search_term'] = training_data['search_term'].replace(dic)

    return training_data

def tokenize_and_stem(text, return_text=False):
    if isinstance(text, str):
        text = text.decode('utf-8')
        stemmer = SnowballStemmer("english")

        # first tokenize by sentence, then by word to ensure that punctuation is caught as it's own token
        tokens = [word for sent in nltk.sent_tokenize(text) for word in nltk.word_tokenize(sent)]

        # filter out any tokens not containing letters (e.g., numeric tokens, raw punctuation)
        stems = [stemmer.stem(t) for t in tokens]

        return " ".join(stems) if return_text else stems
    return text


def common_words(s1, s2):
    words, cnt = s1.split(), 0
    for word in words:
        if s2.find(word) >= 0:
            cnt += 1
    return cnt


def find_occurences(s1, s2):
    return s2.count(s1)


def preprocess_text(text):
    if isinstance(text, str):
    	# text = re.sub(r'[^\x00-\x7f]', r'', text)
        text = text.lower()
        text = text.replace("  ", " ")
        text = text.replace(",", "")
        text = text.replace("$", " ")
        text = text.replace("?", " ")
        text = text.replace("-", " ")
        text = text.replace("//", "/")
        text = text.replace("..", ".")
        text = text.replace(" / ", " ")
        text = text.replace(" \\ ", " ")
        text = text.replace(".", " . ")

        text = re.sub(r"([0-9])( *)\.( *)([0-9])", r"\1.\4", text)
        text = re.sub(r"([0-9]+)( *)(inches|inch|in|')\.?", r"\1in. ", text)
        text = re.sub(r"([0-9]+)( *)(foot|feet|ft|'')\.?", r"\1ft. ", text)
        text = re.sub(r"([0-9]+)( *)(pounds|pound|lbs|lb)\.?", r"\1lb. ", text)
        text = re.sub(r"([0-9]+)( *)(square|sq) ?\.?(feet|foot|ft)\.?", r"\1sq.ft. ", text)
        text = re.sub(r"([0-9]+)( *)(cubic|cu) ?\.?(feet|foot|ft)\.?", r"\1cu.ft. ", text)
        text = re.sub(r"([0-9]+)( *)(gallons|gallon|gal)\.?", r"\1gal. ", text)
        text = re.sub(r"([0-9]+)( *)(ounces|ounce|oz)\.?", r"\1oz. ", text)
        text = re.sub(r"([0-9]+)( *)(centimeters|cm)\.?", r"\1cm. ", text)
        text = re.sub(r"([0-9]+)( *)(milimeters|mm)\.?", r"\1mm. ", text)
        text = text.replace("°", " degrees ")
        text = re.sub(r"([0-9]+)( *)(degrees|degree)\.?", r"\1deg. ", text)
        text = text.replace(" v ", " volts ")
        text = re.sub(r"([0-9]+)( *)(volts|volt)\.?", r"\1volt. ", text)
        text = re.sub(r"([0-9]+)( *)(watts|watt)\.?", r"\1watt. ", text)
        text = re.sub(r"([0-9]+)( *)(amperes|ampere|amps|amp)\.?", r"\1amp. ", text)
    return text


def preprocess_data():
    if os.path.isfile("../../dataset/preprocessed_training_data.csv"):
        print("Found Preprocessed DataFrame")
        return pd.read_csv("../../dataset/preprocessed_training_data.csv", encoding="ISO-8859-1")
    else:
        print("Preprocessing Started")
        print("")

        training_data = pd.read_csv("../../dataset/train.csv", encoding="ISO-8859-1")
        descriptions = pd.read_csv("../../dataset/product_descriptions.csv", encoding="ISO-8859-1")
        attributes = pd.read_csv("../../dataset/attributes.csv")
        brands = attributes[attributes.name == "MFG Brand Name"][["product_uid", "value"]].rename(
            columns={"value": "brand"})

        print("Preprocess Search Terms")
        training_data['search_term'] = training_data['search_term'].map(
            lambda i: tokenize_and_stem(preprocess_text(str(unidecode(i))), return_text=True))

        print("Preprocess Titles")
        training_data['product_title'] = training_data['product_title'].map(
            lambda i: tokenize_and_stem(preprocess_text(str(unidecode(i))), return_text=True))

        print("Preprocess Descriptions")
        descriptions['product_description'] = descriptions['product_description'].map(
            lambda i: tokenize_and_stem(preprocess_text(str(unidecode(i))), return_text=True))

        print(descriptions['product_description'])

        print("Preprocess Brands")
        brands['brand'] = brands['brand'].map(
            lambda i: tokenize_and_stem(preprocess_text(i), return_text=True))

        print("Merge data with descriptions")
        training_data = pd.merge(training_data, descriptions, how='left', on='product_uid')

        print("Merge data with brands")
        training_data = pd.merge(training_data, brands, how='left', on='product_uid')

        training_data = fixtypos(training_data)
        training_data['info'] = training_data['search_term'] + "\t" + training_data['product_title'] + "\t" + \
                                training_data['product_description']

        training_data.to_csv('../../dataset/preprocessed_training_data.csv', encoding='utf-8')

        return training_data


def feature_generation():
    training_data = preprocess_data()

    print(training_data)
    print("")

    print("Creating Feature Dataframe")
    feature_df = pd.DataFrame(
        columns=[
            'search_term_length',
            'search_word_count',
            'title_word_count',
            'desc_word_count',
            'search_text_occurences_in_title',
            'search_text_occurences_in_description',
            'search_last_word_in_title',
            'search_last_word_in_description',
            'search_title_common_words',
            'search_description_common_words',
            # 'brand_word_count',
            # 'search_brand_common_words',
            # 'brand_rate',
            # 'brands_numerical',
        ],
        index=training_data['id'].tolist()
    )

    training_data['attr'] = training_data['search_term'] + "\t" + training_data['brand']
    brands = pd.unique(training_data.brand.ravel())
    d = {}
    i = 1000
    for s in brands:
        d[s] = i
        i += 3

    feature_df['search_term_length'] = training_data['search_term'].map(lambda i: len(i))
    feature_df['search_word_count'] = training_data['search_term'].map(lambda i: len(i.split())).astype(np.int64)
    feature_df['title_word_count'] = training_data['product_title'].map(lambda i: len(i.split())).astype(np.int64)
    feature_df['desc_word_count'] = training_data['product_description'].map(lambda i: len(i.split())).astype(np.int64)
    feature_df['search_text_occurences_in_title'] = training_data['info'].map(
        lambda i: find_occurences(i.split('\t')[0], i.split('\t')[1]))
    feature_df['search_text_occurences_in_description'] = training_data['info'].map(
        lambda i: find_occurences(i.split('\t')[0], i.split('\t')[2]))
    training_data['search_term'].map(lambda i: len(i.split())).astype(np.int64)
    feature_df['search_last_word_in_title'] = training_data['info'].map(
        lambda i: find_occurences(i.split('\t')[0].split(" ")[-1], i.split('\t')[1]))
    feature_df['search_last_word_in_description'] = training_data['info'].map(
        lambda i: find_occurences(i.split('\t')[0].split(" ")[-1], i.split('\t')[2]))
    feature_df['search_title_common_words'] = training_data['info'].map(
        lambda i: common_words(i.split('\t')[0], i.split('\t')[1]))
    feature_df['search_description_common_words'] = training_data['info'].map(
        lambda i: common_words(i.split('\t')[0], i.split('\t')[2]))
    # feature_df['brand_word_count'] = training_data['brand'].map(lambda i: len(i.split())).astype(np.int64)
    # feature_df['search_brand_common_words'] = training_data['attr'].map(
    #     lambda i: common_words(i.split('\t')[0], i.split('\t')[1]))
    # feature_df['brand_rate'] = feature_df['search_brand_common_words'] / feature_df['brand_length']
    # feature_df['brands_numerical'] = training_data['brand'].map(lambda x: d[x])

    print(feature_df.shape)
    print(feature_df.head(10))
    feature_df.to_csv('../../dataset/features.csv', encoding='utf-8')


if __name__ == "__main__":
    # preprocess_data()
    feature_generation()
