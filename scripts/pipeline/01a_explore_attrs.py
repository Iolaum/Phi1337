import pandas as pd
import re
import nltk
import pickle

from collections import Counter
from collections import OrderedDict
from sets import Set


def main():

	attributes = pd.read_csv("../../dataset/attributes.csv")

	attrs_series_not_unique = attributes["name"]
	attrs_series = attributes["name"].unique()
	attrs_ids = attributes["product_uid"].unique()

	unique_brands = attributes['value'][attributes['name'] == 'MFG Brand Name'].unique()
	print len(unique_brands)
	#print('Unique brands '  + unique_brands)
	exit()
	# for att in attrs_series:
	# 	print att

	#print number of attributes
	print len(attrs_series)

	#Print series of attributes
	# for prod_id in prod_ids:
	# 	prod_attributes = attributes.loc[attributes['product_uid'] == prod_id]

	#Print the number of products with features 
	print len(attrs_ids)

	
	all_attribute_names = Counter(attrs_series_not_unique.tolist())
	print all_attribute_names.most_common()[:30]


if __name__ == "__main__":
    main()