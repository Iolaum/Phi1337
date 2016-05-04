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
		features_df = pd.read_csv(preprocessed_path, index_col=0)
		feature_cols = list(features_df.columns.values)
		features_np_arr = np.array(features_df)

		should_add_features = True
	else:
		print("Not Found Preprocessed DataFrame")
		return None

	if should_add_features:
		# score_matrix

		target_column = score_df['relevance']

		score_df = score_df.drop(score_df.columns[[-1]], axis=1)
		print score_df.shape

		# TF_IDF
		tf_idf_df = tf_idf_df.drop(tf_idf_df.columns[[-1]], axis=1)
		print tf_idf_df.shape

		# LSA
		lsa_df = lsa_df.drop(lsa_df.columns[[-1]], axis=1)
		print lsa_df.shape

		features_df = pd.DataFrame(features_np_arr, index=score_df.index, columns=feature_cols)

		print features_df.shape
		print(features_df)

		result = pd.concat([score_df, tf_idf_df, lsa_df, features_df, target_column], axis=1, ignore_index=True)

		print result.shape

		result.to_pickle('../../dataset/score_df_final.pickle')


if __name__ == "__main__":
	concatDataFrames()