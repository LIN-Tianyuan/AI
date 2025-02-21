import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
from sklearn.utils import class_weight
from sklearn.ensemble import GradientBoostingClassifier

# MAC OS: brew install libomp

# Data process
data = pd.read_csv('./data/wine_quality.csv')
print(data.head())
x = data.iloc[:, :-1]
y = data.iloc[:, -1] - 3

# x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y, test_size=0.2)

# pd.concat([x_train, y_train], axis=1).to_csv('wine_quality_train.csv')
# pd.concat([x_test, y_test], axis=1).to_csv('wine_quality_test.csv')

# Get data
train_data = pd.read_csv('wine_quality_train.csv')
test_data = pd.read_csv('wine_quality_test.csv')

x_train = train_data.iloc[:, :-1]
y_train = train_data.iloc[:, -1]
x_test = test_data.iloc[:, :-1]
y_test = test_data.iloc[:, -1]
class_weight = class_weight.compute_sample_weight(class_weight='balanced', y=y_train)

# Model train
model = XGBClassifier(n_estimators=5, objective='multi:softmax')
# model =GradientBoostingClassifier(n_estimators=5)
model.fit(x_train, y_train, sample_weight=class_weight)

y_pre = model.predict(x_test)

print(classification_report(y_test, y_pre))
