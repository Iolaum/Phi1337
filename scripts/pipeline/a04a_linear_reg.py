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

def regression(reg_type, standardize_df, debug=False, save_model=False):
	score_df = pd.read_pickle('../../dataset/score_df_final.pickle')

	# Fill NaN value
	# score_df = score_df.fillna(0.0)

	# The last column is the target
	training_set = np.array(score_df)

	# # Debug
	# print(training_set)

	X = training_set[:, :-1] # grab the first to the col before last column
	Y = training_set[:, -1] # the last col_index

	if standardize_df:
		print("Standardizing...")
		scaler = StandardScaler().fit(X)

		print("Saving standardizer")
		with open("../../dataset/scaler.pickle", 'wb') as handle
			pickle.dump(scaler, handle)

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

	xtr, xts, ytr, yts = train_test_split(X, Y, test_size=0.2, random_state=13)

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

	if reg_type == 'linear':
		print("Regression Type - Linear")
		lin_model = LinearRegression()
	elif reg_type == 'svr':
		print("Regression Type - SVR(RBF)")
		lin_model = SVR()
	elif reg_type == 'rfr':
		print("Regression Type - Random Forest Regressor (RFR)")
		lin_model = RandomForestRegressor(
			n_estimators=14,
			max_features='auto',
			max_depth=6
		)

	lin_model.fit(xtr, ytr)
	# print(lin_model.feature_importances_)

	###
	if save_model:
		# now you can save it to a file
		filename = '../../dataset/model_' + reg_type +  '.pickle'
		with open(filename, 'wb') as f:
			pickle.dump(lin_model, f)
	###

	# Check for overfitting. Predicted the relevance for the training data.
	print("\nError on training set")
	ytr_pred = lin_model.predict(xtr)
	ytr_error = sqrt(mean_squared_error(ytr_pred, ytr))
	print(ytr_error)

	print("")

	# Predicted the relevance for the test data.
	print("Error on validation set. Check for overfitting")
	yts_pred = lin_model.predict(xts)
	yts_error = sqrt(mean_squared_error(yts_pred, yts))

	print(yts_error)


if __name__ == "__main__":
	# Change between:
	# svr
	# linear
	# rfr
	regression_type = 'linear'
	standardize_df = True
	save_model = True

	regression(regression_type, standardize_df, debug=False, save_model=True)
