import matplotlib.pyplot as plt
import pandas as pd
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
import joblib

# 1.Read data
# data = pd.read_csv('number.csv')
# x = data.iloc[:, 1:]
# y = data.iloc[:, 0]
# print(Counter(y))

# 2.Display
# digit = x.iloc[1000].values
# img = digit.reshape(28, 28)
# plt.imshow(img, cmap='gray')
# plt.imsave('digit.png', img)
# plt.show()

# 3.Data preprocessing
# 3.1 Normalization
# x = x / 255.
# # 3.2 Data set segmentation
# x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, stratify=y, random_state=22)
#
# # 4.Model training
# # 4.1 Instantiation
# model = KNeighborsClassifier(n_neighbors=11)
# # 4.2 Train
# model.fit(x_train, y_train)
#
# # 5.Model prediction
# img = plt.imread('digit.png')
# img = img[:, :, 1].reshape(1, -1) / 255.
# y_predict = model.predict(x_test)
# print(y_predict)
#
# # 6.Model Evaluation
# print(model.score(x_test, y_test))
# print(accuracy_score(y_predict, y_test))
#
# # 7.Model Save
# joblib.dump(model, 'knn.pth')

# 8.Model Load
knn = joblib.load('knn.pth')
# print(knn.score(x_test,y_test))
img = plt.imread('digit.png')
img = img[:, :, 1].reshape(1, -1)
print(knn.predict(img))
