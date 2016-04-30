# -*- coding: utf-8 -*-

import numpy as np
import os
import pandas as pd
from nltk.stem.porter import *

from word_count_evaluation import tokenize_and_stem
from word_count_evaluation import fixtypos
from unidecode import unidecode

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

    training_data.to_csv('../../dataset/preprocessed_training_data.csv', encoding='utf-8')

    return training_data


def feature_generation():
    if os.path.isfile("../../dataset/preprocessed_training_data.csv"):
        print("Found Preprocessed DataFrame")
        training_data = pd.read_csv("../../dataset/preprocessed_training_data.csv", encoding="ISO-8859-1")
    else:
        training_data = preprocess_data()

    print(training_data)
    print("")

    print("Creating Feature Dataframe")

    feature_df = pd.DataFrame(
        columns=[
            'search_length',
            'title_length',
            'desc_length',
            'brand_length', \
            'search_text_occurences_in_title',  #
            'search_text_occurences_in_description',
            'search_last_word_in_title',
            'search_last_word_in_description',
            'search_title_common_words',
            'search_description_common_words',
            'search_brand_common_words',
            'brand_rate',
        ],
        index=training_data['id'].tolist()
    )


# df_all['len_of_query'] = df_all['search_term'].map(lambda x:len(x.split())).astype(np.int64)
# df_all['len_of_title'] = df_all['product_title'].map(lambda x:len(x.split())).astype(np.int64)
# df_all['len_of_description'] = df_all['product_description'].map(lambda x:len(x.split())).astype(np.int64)

# df_all['len_of_brand'] = df_all['brand'].map(lambda x:len(x.split())).astype(np.int64)
# df_all['search_term'] = df_all['product_info'].map(lambda x:seg_words(x.split('\t')[0],x.split('\t')[1]))

# #print("--- Search Term Segment: %s minutes ---" % round(((time.time() - start_time)/60),2))
# df_all['query_in_title'] = df_all['product_info'].map(lambda x:str_whole_word(x.split('\t')[0],x.split('\t')[1],0))
# df_all['query_in_description'] = df_all['product_info'].map(lambda x:str_whole_word(x.split('\t')[0],x.split('\t')[2],0))
# print("--- Query In: %s minutes ---" % round(((time.time() - start_time)/60),2))
# df_all['query_last_word_in_title'] = df_all['product_info'].map(lambda x:str_common_word(x.split('\t')[0].split(" ")[-1],x.split('\t')[1]))
# df_all['query_last_word_in_description'] = df_all['product_info'].map(lambda x:str_common_word(x.split('\t')[0].split(" ")[-1],x.split('\t')[2]))
# print("--- Query Last Word In: %s minutes ---" % round(((time.time() - start_time)/60),2))
# df_all['word_in_title'] = df_all['product_info'].map(lambda x:str_common_word(x.split('\t')[0],x.split('\t')[1]))
# df_all['word_in_description'] = df_all['product_info'].map(lambda x:str_common_word(x.split('\t')[0],x.split('\t')[2]))


# df_all['ratio_title'] = df_all['word_in_title']/df_all['len_of_query']
# df_all['ratio_description'] = df_all['word_in_description']/df_all['len_of_query']


# df_all['attr'] = df_all['search_term']+"\t"+df_all['brand']
# df_all['word_in_brand'] = df_all['attr'].map(lambda x:str_common_word(x.split('\t')[0],x.split('\t')[1]))
# df_all['ratio_brand'] = df_all['word_in_brand']/df_all['len_of_brand']
# df_brand = pd.unique(df_all.brand.ravel())
# d={}
# i = 1000
# for s in df_brand:
#     d[s]=i
#     i+=3
# df_all['brand_feature'] = df_all['brand'].map(lambda x:d[x])
# df_all['search_term_feature'] = df_all['search_term'].map(lambda x:len(x))



if __name__ == "__main__":
    # preprocess_data()
    feature_generation()
