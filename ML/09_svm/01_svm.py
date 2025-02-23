from sklearn.datasets import load_iris
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

from plot_util import plot_decision_boundary

# 1.Load data
X, y = load_iris(return_X_y=True)

# print(y.shape)
# print(X.shape)
x = X[y < 2, :2]
y = y[y < 2]
# print(y.shape)

# plt.scatter(x[y == 0, 0], x[y == 0, 1], c='red')
# plt.scatter(x[y == 1, 0], x[y == 1, 1], c='blue')
# plt.show()

# 2.Data preprocessing
transform = StandardScaler()
x_tran = transform.fit_transform(x)

# 3.Model train
model = LinearSVC(C=10)
model.fit(x_tran, y)
y_pred = model.predict(x_tran)

# print(accuracy_score(y_pred, y))

# 4.Visualization
plot_decision_boundary(model, axis=[-3, 3, -3, 3])
plt.scatter(x_tran[y == 0, 0], x_tran[y == 0, 1], c='red')
plt.scatter(x_tran[y == 1, 0], x_tran[y == 1, 1], c='blue')
plt.show()
