# 0.Import package
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import numpy as np

# 1.Load data
data = pd.read_csv('breast-cancer-wisconsin.csv')
# print(data.info())
# 2.Data process
# 2.1 Missing value
data = data.replace(to_replace='?', value=np.NAN)
data = data.dropna()

# 2.2 Get features and target values
X = data.iloc[:, 1:-1]
y = data['Class']

# 2.3 Data segmentation
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=22)
# 3. Feature engineering (standardization)
pre = StandardScaler()
x_train = pre.fit_transform(x_train)
x_test = pre.transform(x_test)
# 4. Model training
model = LogisticRegression()
model.fit(x_train, y_train)

# 5. Model prediction and evaluation
y_predict = model.predict(x_test)
print(y_predict)
print(accuracy_score(y_test, y_predict))
