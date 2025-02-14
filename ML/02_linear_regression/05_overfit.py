"""
Overfit
"""
# 0.Import Toolkit
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

# 1.Preparation of data
np.random.seed(22)
x = np.random.uniform(-3, 3, size=100)
# print(x)
y = 0.5 * x ** 2 + x + 2 + np.random.normal(0, 1, size=100)
# print(y)
# 2.Model training
model = LinearRegression()
X = x.reshape(-1,1)
X3 = np.hstack([X,X**2,X**3,X**4,X**5,X**6,X**7,X**9,X**10])
model.fit(X3,y)

# 3.Prediction
y_predict  = model.predict(X3)
print(mean_squared_error(y_true=y,y_pred=y_predict))
# 4.Demonstration effect
plt.scatter(x,y)
plt.plot(np.sort(x),y_predict[np.argsort(x)])
plt.show()






