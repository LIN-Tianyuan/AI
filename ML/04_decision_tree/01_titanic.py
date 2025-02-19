# 0.Import package
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt

# 1.Read data
data = pd.read_csv('./titanic/train.csv')
# print(data.head())
# print(data.info())

# 2.Data process
x = data[['Pclass', 'Sex', 'Age']].copy()
y = data['Survived'].copy()
# print(x.head(10))
x['Age'].fillna(x['Age'].mean(), inplace=True)
# print(x.head(10))
x = pd.get_dummies(x)
# print(x.head(10))

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

# 3.Model train
model = DecisionTreeClassifier()
model.fit(x_train, y_train)


# 4.Model predict
y_pre = model.predict(x_test)

print(classification_report(y_true=y_test,y_pred=y_pre))

# 5.Visualization
plot_tree(model)
plt.show()


