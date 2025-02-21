import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

# 1.Read data
data = pd.read_csv('./data/titanic/train.csv')
# print(data.head())
# print(data.info())

# 2.Data process
x = data[['Pclass', 'Sex', 'Age']].copy()
y = data['Survived'].copy()
# print(x.head(10))
# Old: x['Age'].fillna(x['Age'].mean(), inplace=True)
x['Age'] = x['Age'].fillna(x['Age'].mean())
# print(x.head(10))
x = pd.get_dummies(x)
# print(x.head(10))

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

model = GradientBoostingClassifier()
model.fit(x_train, y_train)
print(model.score(x_test, y_test))
