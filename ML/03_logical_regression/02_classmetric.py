from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
import pandas as pd

# Constructed data: real values, predicted values
y_true = ['Bad', 'Bad', 'Bad', 'Bad', 'Bad', 'Bad', 'Good', 'Good', 'Good', 'Good']
y_pre_A = ['Bad', 'Bad', 'Bad', 'Good', 'Good', 'Good', 'Good', 'Good', 'Good', 'Good']
y_pre_B = ['Bad', 'Bad', 'Bad', 'Bad', 'Bad', 'Bad', 'Bad', 'Bad', 'Bad', 'Good']

# Confusion matrix
A = confusion_matrix(y_true, y_pre_A, labels=['Bad', 'Good'])
print(pd.DataFrame(A, columns=['Bad', 'Good'], index=['Bad', 'Good']))
B = confusion_matrix(y_true, y_pre_B, labels=['Bad', 'Good'])
print(pd.DataFrame(B, columns=['Bad', 'Good'], index=['Bad', 'Good']))
"""
      Bad  Good
Bad     3     3
Good    0     4
      Bad  Good
Bad     6     0
Good    3     1
"""

# Precision
print(precision_score(y_true, y_pre_A, pos_label='Bad'))
print(precision_score(y_true, y_pre_B, pos_label='Bad'))
"""
1.0
0.6666666666666666
"""

# Recall
print(recall_score(y_true, y_pre_A, pos_label='Bad'))
print(recall_score(y_true, y_pre_B, pos_label='Bad'))
"""
0.5
1.0
"""

# f1-score
print(f1_score(y_true,y_pre_A,pos_label='Bad'))
print(f1_score(y_true,y_pre_B,pos_label='Bad'))
"""
0.6666666666666666
0.8
"""
