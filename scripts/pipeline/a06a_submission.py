import pandas as pd
import numpy as np
import pickle

from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.svm import SVR
from sklearn.svm import LinearSVR

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor

def regression(reg_type, standardize_df, debug=False):
	# load model
	filename = '../../dataset/model_' + reg_type +  '.pickle'
	lin_model = None
	with open(filename, 'rb') as f:
		lin_model = pickle.load(f)

	score_df_tst = pd.read_pickle('../../dataset/score_df_final_tst.pickle')

	# Fill NaN value
	# score_df = score_df.fillna(0.0)

	# The last column is the target
	X = np.array(score_df_tst)

	if standardize_df:
		print("Standardizing...")
		X = StandardScaler().fit_transform(X)

	# Debug

	if debug:
		print("Score DataFrame")
		print(score_df)
		print("")

		print("Training Values")
		print(X)
		print("")

		print("Output Values")
		print(Y)
		print("")

		print("Shapes of X and Y")
		print(X.shape)
		print(Y.shape)

	# Debug
	if debug:
		print("XTR - XTS")
		print(xtr.shape)
		print(xts.shape)
		print("")

		print("YTR - YTS")
		print(ytr.shape)
		print(yts.shape)
		print("")

	yts_pred = lin_model.predict(X)

	#yts_error = sqrt(mean_squared_error(yts_pred, yts))
	print("Prediction by (" + reg_type + ") on Test data have finished")

	# create submission file
	id_series = pd.read_csv('../../dataset/test.csv')['id']
	submission_df = pd.DataFrame(id_series, columns=['id'])
	submission_df['relevance'] = yts_pred
	submission_df.to_csv('../../dataset/submission.csv', columns=['id', 'relevance'])

if __name__ == "__main__":
	# Change between:
	# svr
	# linear
	# rfr
	regression_type = 'rfr'
	standardize_df = True

	regression(regression_type, standardize_df, debug=False)