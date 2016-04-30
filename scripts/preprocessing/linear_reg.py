import pandas as pd
import numpy as np

from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor



def regression(reg_type, use_tfidf, standardize_df, debug=False):
    score_df = pd.read_pickle('../../dataset/score_df.pickle')

    if use_tfidf:
        # tfidf score dataframe
        print("Running Regression with TFIDF score dataframe")
        score_df = pd.read_pickle('../../dataset/score_df_tfidf.pickle')

    training_set = np.array(score_df)
    # # Debug
    # print(training_set)

    X = training_set[:, range(0, 3)]
    Y = training_set[:, 3]

    if standardize_df:
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
    elif reg_type == 'SVR':
        print("Regression Type - SVR(RBF)")
        lin_model = SVR()
    elif reg_type == 'rfr':
        print("Regression Type - Random Forest Regressor (RFR)")
        lin_model = RandomForestRegressor()

    lin_model.fit(xtr, ytr)

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
    # Change between SVR or Linear or rfr
    regression_type = 'rfr'
    standardize_df = False

    # Change to use tfidf scores
    use_tfidf = False

    regression(regression_type, use_tfidf, standardize_df, debug=False)
