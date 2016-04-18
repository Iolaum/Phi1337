
# coding: utf-8

# In[24]:

get_ipython().magic('matplotlib inline')
import numpy as np
import pandas as pd
import io
from subprocess import check_output
import collections


# In[2]:

"Read Files"
training_data = pd.read_csv("train.csv", encoding = "ISO-8859-1")
testing_data = pd.read_csv("test.csv", encoding = "ISO-8859-1")
attribute_data = pd.read_csv("attributes.csv")
descriptions = pd.read_csv("product_descriptions.csv")


# In[3]:

"Merge Descriptions"
training_data = pd.merge(training_data, descriptions,
                         on="product_uid", how="left")


# In[7]:

"Merge Product Counts"
product_counts = pd.DataFrame(pd.Series(training_data.groupby                                        (["product_uid"]).size(),
                                        name="product_count"))
training_data = pd.merge(training_data, product_counts,                         left_on="product_uid", right_index=True,                         how="left")


# In[8]:

"Merge brand Names"
brand_names = attribute_data[attribute_data.name == "MFG Brand Name"][["product_uid", "value"]].rename(columns={"value": "brand_name"})
training_data = pd.merge(training_data, brand_names, on="product_uid",                         how="left")
training_data.brand_name.fillna("Unknown", inplace=True)


# In[9]:

"Check enteries in traning_data after preprocessing"
print(str(training_data.info()))


# In[10]:

"Check basic details in training_data after preprocessing"
print(str(training_data.describe()))


# In[12]:

"Check training_data table"
training_data[:5]


# In[14]:

"Check the value_counts in attribute_data file"
print(attribute_data.name.value_counts())


# In[16]:

"Check attribute_data names"
print(attribute_data.value[attribute_data.name == "indoor/Outdoor"]     .value_counts())


# In[17]:

training_data["in_bins"] = pd.cut(training_data.id, 20, labels=False)
print(training_data.corr(method="spearman"))
training_data.describe()


# In[18]:

"print histogram of training_data"
training_data.relevance.hist()
training_data.relevance.value_counts()


# In[19]:

"Histogram - product_descriptions length"
(descriptions.product_description.str.len() / 5).hist(bins=30)


# In[20]:

"Histogram - product_title"
(training_data.product_title.str.len() / 5).hist(bins=30)


# In[21]:

"Histogram - Search terms, Search Count"
(training_data.search_term.str.len() / 5).hist(bins=30)
(training_data.search_term.str.count("\\s+") + 1).hist(bins=30)


# In[22]:

"Value count - training_data"
testing_data.product_uid.value_counts()


# In[23]:

"Product Cosing Value"
training_products = training_data.product_uid.value_counts()
testing_products = testing_data.product_uid.value_counts()
training_norm = np.sqrt((training_products ** 2).sum())
testing_norm = np.sqrt((testing_products ** 2).sum())
product_uid_cos = (training_products * testing_products).sum()/ (training_norm * testing_norm)
print("Product distribution cosine:", product_uid_cos)


# In[25]:

"bag of words"
chars = collections.Counter()
for title in training_data.product_title:
    chars.update(title.lower())
total = sum(chars.values())

print("Title char counts")
for c, count in chars.most_common(30):
    print("0x{:02x} {}: {:.1f}%".format(ord(c),  c, 100. * count / total))
    
words = collections.Counter()
for title in training_data.search_term:
    words.update(title.lower().split())

total = sum(words.values())
print("Search word counts")
for word, count in words.most_common(200):
    print("{}: {:.1f}% ({:,})".format(word, 100. * count / total, count))


# In[26]:

print("Indoor/outdoor", training_data.search_term.str.contains      ("indoor|outdoor|interior|exterior", case=False).value_counts())
print("Contains numbers", training_data.search_term.str.contains      ("\\d", case=False).value_counts())


# In[27]:

"Summarize Values - bag of words - Category-wise"
def summarize_values(name, values):
    values.fillna("", inplace=True)
    counts = collections.Counter()
    for value in values:
        counts[value.lower()] += 1
    
    total = sum(counts.values())
    print("{} counts ({:,} values)".format(name, total))
    for word, count in counts.most_common(20):
        print("{}: {:.1f}% ({:,})".format(word, 100. * count / total,                                          count))

for attribute_name in ["Color Family", "Color/Finish", "Material",                       "MFG Brand Name", "Indoor/Outdoor", "Commercial                       / Residential"]:
    summarize_values("\n" + attribute_name, attribute_data                     [attribute_data.name == attribute_name].value)


# In[ ]:




# In[ ]:



