import pandas as pd
import numpy as np

from sklearn.cross_validation import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from math import sqrt

score_df = pd.read_pickle('../../dataset/score_df.pickle')
training_set = np.array(score_df)

X = training_set[:, range(1, 4)]
Y = training_set[:, 4]

# # Debug
# print(X)
# print(Y)
# print(X.shape)
# print(Y.shape)

xtr, xts, ytr, yts = train_test_split(X, Y, test_size=0.2, random_state=13)

# # Debug
# print(xtr.shape)
# print(xts.shape)
# print(ytr.shape)
# print(yts.shape)

lin_model = LinearRegression()
lin_model.fit(xtr, ytr)

ytr_pred = lin_model.predict(xtr)
ytr_error = sqrt(mean_squared_error(ytr_pred, ytr))

yts_pred = lin_model.predict(xts)
yts_error = sqrt(mean_squared_error(yts_pred, yts))

print(ytr_error)
print(yts_error)
