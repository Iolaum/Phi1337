
# coding: utf-8

# In[6]:

import numpy as np
import pandas as pd
import nltk
from nltk.collocations import *
import scipy
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.cross_validation import train_test_split
from sklearn import linear_model
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')


# In[4]:

train_df = pd.read_csv("train.csv", encoding="ISO-8859-1")
train_df["search_term"] = train_df["product_title"].str.lower()
train_df["product_title"] = train_df["product_title"].str.lower()


# In[5]:

des=pd.read_csv("product_descriptions.csv",encoding="ISO-8859-1")
des["product_description"] = des["product_description"].str.lower()
train_df = train_df.merge(des, on="product_uid")
train_df = train_df.assign(prod_complete = lambda x:                          (x['product_title'] + ' ' +                           ['product_description']))


# In[8]:

s = train_df["relevance"]
notrelevant = train_df[s==1.00]
relevant = train_df[s==3.00]
s.hist()


# In[10]:

vectorizer = CountVectorizer(ngram_range=(1, 3), min_df=1)
search_counts = vectorizer.fit_transform(train_df["search_term"])
distinct_title_counts = vectorizer.transform(train_df["product_title"].                                             drop_duplicates())
distinct_descr_counts = vectorizer.transform(
    des["product_description"].drop_duplicates())
feature_counts = scipy.sparse.vstack(
    [search_counts,distinct_title_counts,distinct_descr_counts])


# In[11]:

notrelevant.sample(n=5)


# In[12]:

nrl = notrelevant[notrelevant["id"] == 24058]
nrl


# In[13]:

train_df[train_df["product_title"].str.contains("hydronic")]


# In[15]:

prods =  pd.read_csv("product_descriptions.csv", encoding="ISO-8859-1")
prods.sample(n=5)


# In[16]:

hydronics = prods[prods["product_description"].str.contains("hydronic")]
hydronics[hydronics["product_description"].str.contains("heater")]


# In[17]:

bm = nltk.collocations.BigramAssocMeasures()
finder = BigramCollocationFinder.from_words(prods["product_description"].str.cat())


# In[18]:

score = finder.score_ngram(bm.pmi,"hydronic","heater")
score


# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:




# In[ ]:



