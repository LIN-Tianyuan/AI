# 0.Import package
# from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.metrics import mean_squared_error

"""
# 1.Load data   `load_boston` has been removed from scikit-learn since version 1.2.
boston = load_boston()
print(boston)

# 2.Data set segmentation
# x_train,x_test,y_train,y_test =train_test_split(boston.data,boston.target,test_size=0.2,random_state=22)
"""
import pandas as pd
import numpy as np

# 1.Load data
data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
target = raw_df.values[1::2, 2]

# 2.Data set segmentation
x_train, x_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=22)

# 3.Standardization
process = StandardScaler()
x_train = process.fit_transform(x_train)
x_test = process.transform(x_test)

# 4.Model training
# 4.1 Instantiation (regular equations)
# model = LinearRegression(fit_intercept=True)
# 4.1 Instantiation (gradient descent)
model = SGDRegressor(learning_rate='constant', eta0=0.01)
# 4.2 fit
model.fit(x_train, y_train)

# print(model.coef_)
# print(model.intercept_)
# 5.Prediction
y_predict = model.predict(x_test)

print(y_predict)

# 6.Model Evaluation
print(mean_squared_error(y_test, y_predict))
