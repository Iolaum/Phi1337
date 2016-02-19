# coding: utf-8

"defining processing functions"
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor
from nltk.stem.snowball import SnowballStemmer

# For the stemming we are using the Snowball Stemmer
stemmer = SnowballStemmer('english')

df_train = pd.read_csv('../dataset/train.csv', encoding="ISO-8859-1")  # read train.csv
df_test = pd.read_csv('../dataset/test.csv', encoding="ISO-8859-1")  # read test.csv
df_pro_desc = pd.read_csv('../dataset/product_descriptions.csv')  # read product_description.csv

num_train = df_train.shape[0]

print("defining processing functions...")


def str_stemmer(s):
    return " ".join([stemmer.stem(word) for word in s.lower().split()])


def str_common_word(str1, str2):
    return sum(int(str2.find(word) >= 0) for word in str1.split())


def str_common_letter(str1, str2):
    return sum(int(str2.find(letter) >= 0) for letter in str1)


"process data"
print("process data...")

print("Concatenating train and test data")
df_all = pd.concat((df_train, df_test), axis=0, ignore_index=True)  # concat train and test data

print("Merging concatenated data with product description")
df_all = pd.merge(df_all, df_pro_desc, how='left', on='product_uid')  # merge concatenated data with product description

print("Stemmming on search terms...")
df_all['search_term'] = df_all['search_term'].map(lambda x: str_stemmer(x))

print("Stemmming on Product titles...")
df_all['product_title'] = df_all['product_title'].map(lambda x: str_stemmer(x))

print("Stemmming on Product descriptions.....")
df_all['product_description'] = df_all['product_description'].map(lambda x: str_stemmer(x))

print("Convert type of Length of query to int64...")
df_all['len_of_query'] = df_all['search_term'].map(lambda x: len(x.split())).astype(np.int64)

print("Concatenate search term with product title and product description...")
df_all['product_info'] = df_all['search_term'] + "\t" + df_all['product_title'] + "\t" + df_all['product_description']

print("Calculate the common words in the product info...")
df_all['word_in_title'] = df_all['product_info'].map(lambda x: str_common_word(x.split('\t')[0], x.split('\t')[1]))

print("Calculate the common words from product description...")
df_all['word_in_description'] = df_all['product_info'].map(
        lambda x: str_common_word(x.split('\t')[0], x.split('\t')[2]))

print("Letters in title...??")
df_all['letter_in_title'] = df_all['product_info'].map(lambda x: str_common_letter(x.split('\t')[0], x.split('\t')[1]))

print("Letters in description...??")
df_all['letter_in_description'] = df_all['product_info'].map(
        lambda x: str_common_letter(x.split('\t')[0], x.split('\t')[2]))

print("Drop columns that were changed...")
df_all = df_all.drop(['search_term', 'product_title', 'product_description', 'product_info'], axis=1)

# Set up training and test sets
df_train = df_all.iloc[:num_train]
df_test = df_all.iloc[num_train:]

id_test = df_test['id']
y_train = df_train['relevance'].values

# Drop 'id' and 'relevance' columns from the training and test sets
X_train = df_train.drop(['id', 'relevance'], axis=1).values
X_test = df_test.drop(['id', 'relevance'], axis=1).values

# Setup RandomForest and Bagging Regressors
rf = RandomForestRegressor(n_estimators=15, max_depth=6, random_state=0)
clf = BaggingRegressor(rf, n_estimators=45, max_samples=0.1, random_state=25)

# Fit the training data into the regression model using the output values
clf.fit(X_train, y_train)

# Run the prediction
y_pred = clf.predict(X_test)

# Set up our Data Frame
datafr = pd.DataFrame({"id": id_test, "relevance": y_pred}).to_csv('../dataset/submission.csv', index=False)
print(datafr)
