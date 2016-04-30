import pandas as pd
import numpy as np

from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.svm import SVR


def regression(regression_type, use_tfidf, debug=False):
    score_df = pd.read_pickle('../../dataset/score_df.pickle')
    rate_range = range(0, 4)
    y_col = 4

    if use_tfidf:
        # tfidf score dataframe
        print("Running Regression with TFIDF score dataframe")
        score_df = pd.read_pickle('../../dataset/score_df_tfidf.pickle')

        rate_range = range(0, 3)
        y_col = 3

    training_set = np.array(score_df)
    # # Debug
    # print(training_set)

    X = training_set[:, rate_range]
    Y = training_set[:, y_col]

    # Debug

    if debug:
        print(score_df)
        print(X)
        print(Y)
        print(X.shape)
        print(Y.shape)

    xtr, xts, ytr, yts = train_test_split(X, Y, test_size=0.2, random_state=13)

    # Debug
    if debug:
        print(xtr.shape)
        print(xts.shape)
        print(ytr.shape)
        print(yts.shape)

    if regression_type == 'linear':
        print("Regression Type - Linear")
        lin_model = LinearRegression()
    else:
        print("Regression Type - SVR(RBF)")
        lin_model = SVR()

    lin_model.fit(xtr, ytr)

    # Check for overfitting. Predicted the relevance for the training data.
    print("")
    print("Check for overfitting")
    ytr_pred = lin_model.predict(xtr)
    ytr_error = sqrt(mean_squared_error(ytr_pred, ytr))
    print(ytr_error)

    print("")

    # Predicted the relevance for the test data.
    print("Predict the relevance for the test data")
    yts_pred = lin_model.predict(xts)
    yts_error = sqrt(mean_squared_error(yts_pred, yts))

    print(yts_error)


if __name__ == "__main__":
    # Change between SVR or Linear
    regression_type = 'svr'

    # Change to use tfidf scores
    use_tfidf = True

    regression(regression_type, use_tfidf, debug=True)
