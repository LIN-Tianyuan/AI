"""
L2 regularization
"""

import numpy as np
from sklearn.linear_model import Ridge
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error


np.random.seed(22)
x = np.random.uniform(-3, 3, size=100)

y = 0.5 * x ** 2 + x + 2 + np.random.normal(0, 1, size=100)


model = Ridge(alpha=0.1)
X = x.reshape(-1,1)
X3 = np.hstack([X,X**2,X**3,X**4,X**5,X**6,X**7,X**9,X**10])
model.fit(X3,y)
print(model.coef_)


y_predict  = model.predict(X3)
print(mean_squared_error(y_true=y,y_pred=y_predict))

plt.scatter(x,y)
plt.plot(np.sort(x),y_predict[np.argsort(x)])
plt.show()