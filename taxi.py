import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os

data = pd.read_csv('../input/train.csv' , nrows = 10_000_000)

data.dtypes

target  = data.fare_amount

train = data.drop(['pickup_datetime', 'dropoff_longitude', 'dropoff_latitude', 'fare_amount'], axis=1)

train = train.drop(['key'], axis =1)


from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score
from sklearn.metrics import mean_squared_error
from math import sqrt

X_train, X_test, y_train, y_test = train_test_split(train, target, test_size = 1/3, random_state = 0)

regressor = LinearRegression()

regressor.fit(X_train,y_train)

y_pred = regressor.predict(X_test)

RMSE_linreg = sqrt(mean_squared_error(y_test, y_pred))

print (RMSE_linreg)

test = pd.read_csv('../input/test.csv' , nrows = 10_000_000)


test_data = test.drop(['key', 'pickup_datetime','dropoff_longitude', 'dropoff_latitude'], axis = 1)

predictions = regressor.predict(test_data)

submission = pd.DataFrame({ 
"key": test["key"], 
"fare_amount": predictions 
})
submission.to_csv("sample-submission-xgb.csv", index=False)
