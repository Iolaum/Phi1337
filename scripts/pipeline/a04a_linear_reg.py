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
import matplotlib.pyplot as plt

import operator

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

	x_labels = ['Title Ratio', 'Desc Ratio', 'Attr Ratio',
				'Title Ratio - TFIDF', 'Desc Ratio - TFIDF', 'Attr Ratio - TFIDF',
				'LSA 1st', 'LSA 2nd', 'LSA 3rd', 'LSA 4th', 
				'Search Length', 'Word Count - Search', 'Word Count - Title', 'Word Count - Desc', 
				'Search in Title', 'Search in Desc', 
				'Last Search Word in Title', 'Last Search Word in Desc', 'Common Words - Search/Title', 
				'Common Words - Search/Desc', 'Word Count - Brand', 'Common Words - Search/Brand', 
				'Brand Ratio', 'Brand - Numerical']
	lin_model.fit(xtr, ytr)
	fitted_features = lin_model.feature_importances_.tolist()

	sorted_features = {}
	for index, value in enumerate(x_labels):
		sorted_features[value] = fitted_features[index]
	sorted_keys = sorted(sorted_features.items(), key=operator.itemgetter(1), reverse=True)
	print sorted_keys
	xx = []
	yy = []
	n = 0
	for index, values in enumerate(sorted_keys):
		n += 1
		xx.append(values[0])
		yy.append(values[1])

		if n == 10:
			break

	X = np.arange(n)

	plt.barh(X, yy, facecolor='#9999ff', align='center', edgecolor='white')
	plt.yticks(X, xx)

	for x, score in zip(X, yy):
		plt.text(score + 0.03, x, '%.3f' % score, ha='center', va='bottom')
	#plt.ylabel('Importance Scores')
	#plt.xlabel('Features name')
	plt.title('Top Ten Features Chosen by Random-Forest')
	plt.show()

	exit()

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
	debug = False

	regression(regression_type, standardize_df, debug=False, save_model=True)

