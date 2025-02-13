# 0.Import Toolkit
from sklearn.datasets import load_iris
# import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# 1.Load Dataset
iris_data = load_iris()
# print(iris_data)
# print(iris_data.target)


# 2.Data Display
iris_df = pd.DataFrame(iris_data['data'], columns=iris_data.feature_names)
iris_df['label'] = iris_data.target
# print(iris_data.feature_names)
# sns.lmplot(x='sepal length (cm)',y='sepal width (cm)',data = iris_df,hue='label')
# plt.show()
print(iris_df)


# 3.Feature engineering (preprocessing - standardization)
# 3.1 Data set segmentation
x_train, x_test, y_train, y_test = train_test_split(iris_data.data, iris_data.target, test_size=0.3, random_state=22)
print(len(iris_data.data))
print(len(x_train))
# 3.2 Standardization
process = StandardScaler()
x_train = process.fit_transform(x_train)
x_test = process.transform(x_test)
# 4.Model training
# 4.1 Instantiation
model = KNeighborsClassifier(n_neighbors=3)
# 4.2 Call the fit method
model.fit(x_train, y_train)
# 5.Model prediction
x = [[5.1, 3.5, 1.4, 0.2]]
x = process.transform(x)
y_predict = model.predict(x_test)
p = model.predict(x)
print(model.predict_proba(x))

# 6. Model evaluation (accuracy)
# 6.1 Using Predicted Results
acc = accuracy_score(y_test, y_predict)
print(acc)

# 6.2 Direct calculation
acc = model.score(x_test, y_test)
print(acc)
