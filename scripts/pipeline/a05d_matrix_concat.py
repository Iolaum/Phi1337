import pandas as pd
import numpy as np

import os

def concatDataFrames():
	score_df = pd.read_pickle('../../dataset/score_df_tst.pickle')
	tf_idf_df = pd.read_pickle('../../dataset/score_df_tfidf_tst.pickle')
	lsa_df = pd.read_pickle('../../dataset/score_df_lsa_cvect_tst.pickle')
	# Read additional features from the result of feature_engineering
	# and append to score_df before saving it.
	# Read from file
	preprocessed_path = '../../dataset/features_t.csv'
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
		features_df = pd.DataFrame(features_np_arr, index=score_df.index, columns=feature_cols)

		result = pd.concat([score_df, tf_idf_df, lsa_df, features_df, target_column], axis=1, ignore_index=True)

		print result.shape

		result.to_pickle('../../dataset/score_df_final_tst.pickle')


if __name__ == "__main__":
	concatDataFrames()