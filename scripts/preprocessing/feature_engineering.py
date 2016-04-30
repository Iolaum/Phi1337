import pandas as pd
import numpy as np

from word_count_evaluation import clean_text, tokenize_and_stem


def feature_generation():
    training_data = pd.read_csv("../../dataset/train.csv", encoding="ISO-8859-1")
    descriptions = pd.read_csv("../../dataset/product_descriptions.csv")
    attributes = pd.read_csv("../../dataset/attributes.csv")



def preprocess_terms():
    import time
start_time = time.time()

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
#from sklearn import pipeline, model_selection
from sklearn import pipeline, grid_search
#from sklearn.feature_extraction import DictVectorizer
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion
from sklearn.decomposition import TruncatedSVD
#from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import mean_squared_error, make_scorer
#from nltk.metrics import edit_distance
from nltk.stem.porter import *
stemmer = PorterStemmer()
#from nltk.stem.snowball import SnowballStemmer #0.003 improvement but takes twice as long as PorterStemmer
#stemmer = SnowballStemmer('english')
import re
#import enchant
import random
random.seed(2016)

df_train = pd.read_csv('../input/train.csv', encoding="ISO-8859-1")[:1000] #update here
df_test = pd.read_csv('../input/test.csv', encoding="ISO-8859-1")[:1000] #update here
df_pro_desc = pd.read_csv('../input/product_descriptions.csv')[:1000] #update here
df_attr = pd.read_csv('../input/attributes.csv')
df_brand = df_attr[df_attr.name == "MFG Brand Name"][["product_uid", "value"]].rename(columns={"value": "brand"})
num_train = df_train.shape[0]
df_all = pd.concat((df_train, df_test), axis=0, ignore_index=True)
df_all = pd.merge(df_all, df_pro_desc, how='left', on='product_uid')
df_all = pd.merge(df_all, df_brand, how='left', on='product_uid')
print("--- Files Loaded: %s minutes ---" % round(((time.time() - start_time)/60),2))

stop_w = ['for', 'xbi', 'and', 'in', 'th','on','sku','with','what','from','that','less','er','ing'] #'electr','paint','pipe','light','kitchen','wood','outdoor','door','bathroom'
strNum = {'zero':0,'one':1,'two':2,'three':3,'four':4,'five':5,'six':6,'seven':7,'eight':8,'nine':9}

def str_stem(s):
        s = re.sub(r"(\w)\.([A-Z])", r"\1 \2", s) #Split words with a.A
        s = s.lower()
        s = s.replace("  "," ")
        s = s.replace(",","") #could be number / segment later
        s = s.replace("$"," ")
        s = s.replace("?"," ")
        s = s.replace("-"," ")
        s = s.replace("//","/")
        s = s.replace("..",".")
        s = s.replace(" / "," ")
        s = s.replace(" \\ "," ")
        s = s.replace("."," . ")
    s = re.sub(r"([0-9])( *)\.( *)([0-9])", r"\1.\4", s)
    s = re.sub(r"([0-9]+)( *)(inches|inch|in|')\.?", r"\1in. ", s)
    s = re.sub(r"([0-9]+)( *)(foot|feet|ft|'')\.?", r"\1ft. ", s)
    s = re.sub(r"([0-9]+)( *)(pounds|pound|lbs|lb)\.?", r"\1lb. ", s)
    s = re.sub(r"([0-9]+)( *)(square|sq) ?\.?(feet|foot|ft)\.?", r"\1sq.ft. ", s)
    s = re.sub(r"([0-9]+)( *)(cubic|cu) ?\.?(feet|foot|ft)\.?", r"\1cu.ft. ", s)
    s = re.sub(r"([0-9]+)( *)(gallons|gallon|gal)\.?", r"\1gal. ", s)
    s = re.sub(r"([0-9]+)( *)(ounces|ounce|oz)\.?", r"\1oz. ", s)
    s = re.sub(r"([0-9]+)( *)(centimeters|cm)\.?", r"\1cm. ", s)
    s = re.sub(r"([0-9]+)( *)(milimeters|mm)\.?", r"\1mm. ", s)
    s = s.replace("°"," degrees ")
    s = re.sub(r"([0-9]+)( *)(degrees|degree)\.?", r"\1deg. ", s)
    s = s.replace(" v "," volts ")
    s = re.sub(r"([0-9]+)( *)(volts|volt)\.?", r"\1volt. ", s)
    s = re.sub(r"([0-9]+)( *)(watts|watt)\.?", r"\1watt. ", s)
    s = re.sub(r"([0-9]+)( *)(amperes|ampere|amps|amp)\.?", r"\1amp. ", s)

    # df_all['search_term'] = df_all['search_term'].map(lambda x:str_stem(x))
    # df_all['product_title'] = df_all['product_title'].map(lambda x:str_stem(x))
    # df_all['product_description'] = df_all['product_description'].map(lambda x:str_stem(x))
    # df_all['brand'] = df_all['brand'].map(lambda x:str_stem(x))
    # print("--- Stemming: %s minutes ---" % round(((time.time() - start_time)/60),2))
    # df_all['product_info'] = df_all['search_term']+"\t"+df_all['product_title'] +"\t"+df_all['product_description']
    # print("--- Prod Info: %s minutes ---" % round(((time.time() - start_time)/60),2))
    # df_all['len_of_query'] = df_all['search_term'].map(lambda x:len(x.split())).astype(np.int64)
    # df_all['len_of_title'] = df_all['product_title'].map(lambda x:len(x.split())).astype(np.int64)
    # df_all['len_of_description'] = df_all['product_description'].map(lambda x:len(x.split())).astype(np.int64)
    # df_all['len_of_brand'] = df_all['brand'].map(lambda x:len(x.split())).astype(np.int64)
    # print("--- Len of: %s minutes ---" % round(((time.time() - start_time)/60),2))
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
    feature_generation(debug=True)
