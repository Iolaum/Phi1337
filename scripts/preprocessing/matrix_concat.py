import pandas as pd
import numpy as np

import os

def concatDataFrames():
	score_df = pd.read_pickle('../../dataset/score_df.pickle')
	tf_idf_df = pd.read_pickle('../../dataset/score_df_tfidf.pickle')
	lsa_df = pd.read_pickle('../../dataset/score_df_lsa_cvect.pickle')
	# Read additional features from the result of feature_engineering
	# and append to score_df before saving it.
	# Read from file
	preprocessed_path = '../../dataset/features.csv'
	features_df = None
	should_add_features = False
	if os.path.isfile(preprocessed_path):
		print("Found Preprocessed DataFrame... Begin appending features to score matrix")
		features_df = pd.read_csv(preprocessed_path)
		should_add_features = True
	else:
		print("Not Found Preprocessed DataFrame")
		exit()

	if should_add_features:
		# score_matrix
		target_column = score_df[-1:]
		score_df = score_df.drop(df.columns[[-1]], axis=1)
		
		result = pd.concat([score_df, features_df, target_column], axis=1)
		print result.shape
		result.to_pickle('../../dataset/score_df2.pickle')

		# TF_IDF
		target_column = tf_idf_df[-1:]
		tf_idf_df = tf_idf_df.drop(df.columns[[-1]], axis=1)
		
		result = pd.concat([tf_idf_df, features_df, target_column], axis=1)
		print result.shape
		result.to_pickle('../../dataset/score_df_tfidf2.pickle')

		# LSA
		target_column = lsa_df[-1:]
		lsa_df = lsa_df.drop(df.columns[[-1]], axis=1)
		
		result = pd.concat([lsa_df, features_df, target_column], axis=1)
		print result.shape
		result.to_pickle('../../dataset/score_df_lsa_cvect2.pickle')


if __name__ == "__main__":
	concatDataFrames()