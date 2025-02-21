# Import package
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import train_test_split

# Read data
data = pd.read_csv('./data/wine0501.csv')
# print(data.info())
data = data[data['Class label'] != 1]
x = data[['Alcohol', 'Hue']].copy()
y = data['Class label'].copy()
# print(y)

pre = LabelEncoder()
y = pre.fit_transform(y)
# print(y)

x_train, x_test, y_train, y_test = train_test_split(x, y)

# Model train
ada = AdaBoostClassifier()
ada.fit(x_train, y_train)

# Model evaluation
print(ada.score(x_test, y_test))
