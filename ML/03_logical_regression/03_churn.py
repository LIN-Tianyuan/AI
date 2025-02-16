# 0.Import package
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

# 1.Data process
data = pd.read_csv('churn.csv')
# print(data.info())
# print(data.head())
data = pd.get_dummies(data)  # Convert category data to numeric data
# print(data.head())
data = data.drop(['Churn_No', 'gender_Male'], axis=1)
# print(data.head())
data = data.rename(columns={'Churn_Yes': 'flag'})
# print(data.head())
# print(data.flag.value_counts())


# 2.Feature engineering
sns.countplot(data=data, y='Contract_Month', hue='flag')
plt.show()

x = data[['PaymentElectronic', 'Contract_Month', 'internet_other']]
y = data['flag']

x_train, x_test, y_train, y_test = train_test_split(x, y, stratify=y, test_size=0.2, random_state=22)

# 3.Model training
LR = LogisticRegression()
LR.fit(x_train, y_train)

# 4.Model prediction and evaluation
y_predict = LR.predict(x_test)
print(accuracy_score(y_test, y_predict))
print(roc_auc_score(y_test, y_predict))
print(classification_report(y_test, y_predict))

"""
0.7877927608232789
0.6779637810328347
              precision    recall  f1-score   support

       False       0.82      0.91      0.86      1035
        True       0.65      0.44      0.53       374

    accuracy                           0.79      1409
   macro avg       0.73      0.68      0.69      1409
weighted avg       0.77      0.79      0.77      1409
"""
