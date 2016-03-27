
# coding: utf-8

# In[3]:

import sqlite3
import pandas as pd


# In[9]:

c = sqlite3.connect(":memory:")


# In[10]:

"Open product_description.csv and convert to sqlite"
df_pd = pd.read_csv("product_descriptions.csv").fillna(" ")
df_pd.to_sql("pd",c)
df_pd = ""


# In[11]:

"Open attributes.csv and convert to sqlite"
df_at = pd.read_csv("attributes.csv").fillna(" ")
df_at.to_sql("at",c)
df_at = ""


# In[12]:

"Open train.csv and convert to sqlite"
df_tr = pd.read_csv("train.csv",
                   sep = ",",
                   encoding = "ISO-8859-1").fillna(" ")
df_tr.to_sql("tr",c)
df_tr = ""


# In[13]:

"Open test.csv and convert to sqlite"
df_te = pd.read_csv("test.csv",
                   sep = ",",
                   encoding = "ISO-8859-1").fillna(" ")
df_te.to_sql("te",c)
df_te = ""


# In[19]:

print("Attribute ------------> Brand")
print(pd.read_sql("SELECT [value], count(*) as c FROM at WHERE[name]='MFG Brand Name' GROUP BY [value]ORDER BY c DESC LIMIT 10;", c).head(10))


# In[22]:

print("Top 10 Search Terms from Train")
print(pd.read_sql("SELECT search_term, count(*) as c FROM tr GROUP BY search_term ORDER BY c DESC LIMIT 10;", c).head(10))


# In[23]:

print("Top 10 Search Terms from Test")
print(pd.read_sql("SELECT search_term, count(*) as c FROM te GROUP BY search_term ORDER BY c DESC LIMIT 10;", c).head(10))


# In[26]:

"Perfect Match from Training data, product_title against search terms"
pd_tr = pd.read_sql("SELECT * FROM tr WHERE relevance='3' LIMIT 1;", c)
print(pd_tr.head())


# In[27]:

"Perfect Match from Training data, product_title"
print(pd.read_sql("SELECT * FROM pd WHERE product_uid="                  + str(pd_tr.product_uid[0]) + ";", c).head())


# In[28]:

"Perfect Match from Training data, product, search and relevance"
print(pd.read_sql("SELECT * FROM at WHERE product_uid='"                  + str(pd_tr.product_uid[0]) + ".0';", c).head(20))


# In[29]:

"Partial Match from Training data, product_title against search terms"
pd_tr = pd.read_sql("SELECT * FROM tr WHERE relevance='2' LIMIT 1;", c)
print(pd_tr.head())


# In[30]:

"Partial Match from Training data, product_title"
print(pd.read_sql("SELECT * FROM pd WHERE product_uid="                  + str(pd_tr.product_uid[0]) + ";", c).head())


# In[31]:

"Partial Match from Training data, product, search and relevance"
print(pd.read_sql("SELECT * FROM at WHERE product_uid='"                  + str(pd_tr.product_uid[0]) + ".0';", c).head(20))


# In[32]:

"Irrelevant Match from Training data, product_title against search terms"
pd_tr = pd.read_sql("SELECT * FROM tr WHERE relevance='1' LIMIT 1;", c)
print(pd_tr.head())


# In[33]:

"Irrelevant Match from Training data, product_title"
print(pd.read_sql("SELECT * FROM pd WHERE product_uid="                  + str(pd_tr.product_uid[0]) + ";", c).head())


# In[34]:

"Irrelevant Match from Training data, product, search and relevance"
print(pd.read_sql("SELECT * FROM at WHERE product_uid='"                  + str(pd_tr.product_uid[0]) + ".0';", c).head(20))


# In[35]:

"Between Partial & Irrelevant Match from Training data"
"product_title against search terms"
pd_tr = pd.read_sql("SELECT * FROM tr WHERE relevance>1 and relevance<2 LIMIT 1;", c)
print(pd_tr.head())


# In[36]:

"Between Partial & Irrelevant Match from Training data, product_title"
print(pd.read_sql("SELECT * FROM pd WHERE product_uid="                   + str(pd_tr.product_uid[0]) + ";", c).head())


# In[38]:

"Between Partial & Irrelevant Match from Training data"
"product, search and relevance"
print(pd.read_sql("SELECT * FROM at WHERE product_uid='"                   + str(pd_tr.product_uid[0]) + ".0';", c).head(20))


# In[ ]:



