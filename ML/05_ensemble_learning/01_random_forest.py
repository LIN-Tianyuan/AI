# 0.Import package
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
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
# Old: x['Age'].fillna(x['Age'].mean(),inplace = True)
x['Age'] = x['Age'].fillna(x['Age'].mean())
# print(x.head(10))
x = pd.get_dummies(x)
print(x.head(10))

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# 3.Model train
# 3.1 Decision tree
tree = DecisionTreeClassifier()
tree.fit(x_train, y_train)

# 3.2 Random Forest
rf = RandomForestClassifier()
rf.fit(x_train, y_train)

# 3.3 Grid search cross validation
# params = {'n_estimators': [10, 20], 'max_depth': [2, 3, 4, 5]}
# model = GridSearchCV(estimator=rf, param_grid=params, cv=3)
# model.fit(x_train, y_train)
# print(model.best_estimator_)    # RandomForestClassifier(max_depth=4, n_estimators=20)

rfs = RandomForestClassifier(max_depth=4, n_estimators=10)
rfs.fit(x_train,y_train)

# 4.Model evaluation

# 4.1 Decision Tree
print(tree.score(x_test,y_test))

# 4.2 Random Forest
print(rf.score(x_test,y_test))


# 4.3 Grid search cross validation
print(rfs.score(x_test,y_test))
