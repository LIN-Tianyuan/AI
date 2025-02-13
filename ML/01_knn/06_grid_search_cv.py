# 0.Import Toolkit
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# 1.Load Dataset
data = load_iris()

# 2.Data Display
x_train, x_test, y_train, y_test = train_test_split(data.data, data.target, test_size=0.2, random_state=22)

# 3.Feature preprocessing
pre = StandardScaler()
x_train = pre.fit_transform(x_train)
x_test = pre.transform(x_test)

# 4.Model instantiation + cross-validation + grid search
model = KNeighborsClassifier(n_neighbors=1)
paras_grid = {'n_neighbors': [4, 5, 7, 9]}
# estimator =GridSearchCV(estimator=model,param_grid=paras_grid,cv=4)
# estimator.fit(x_train,y_train)

# print(estimator.best_score_)
# print(estimator.best_estimator_)
# print(estimator.cv_results_)

"""
0.9666666666666668
KNeighborsClassifier(n_neighbors=7)
{'mean_fit_time': array([0.00127816, 0.00049073, 0.00050187, 0.00043708]), 'std_fit_time': array([1.17588460e-03, 2.18646704e-05, 7.93380587e-05, 4.17717158e-05]), 'mean_score_time': array([0.00344867, 0.00239307, 0.00239134, 0.00228697]), 'std_score_time': array([0.00103811, 0.00013585, 0.00019252, 0.00020572]), 'param_n_neighbors': masked_array(data=[4, 5, 7, 9],
             mask=[False, False, False, False],
       fill_value=999999), 'params': [{'n_neighbors': 4}, {'n_neighbors': 5}, {'n_neighbors': 7}, {'n_neighbors': 9}], 'split0_test_score': array([1., 1., 1., 1.]), 'split1_test_score': array([0.96666667, 0.96666667, 0.96666667, 0.96666667]), 'split2_test_score': array([0.9       , 0.93333333, 0.93333333, 0.93333333]), 'split3_test_score': array([0.9       , 0.93333333, 0.96666667, 0.93333333]), 'mean_test_score': array([0.94166667, 0.95833333, 0.96666667, 0.95833333]), 'std_test_score': array([0.04330127, 0.02763854, 0.02357023, 0.02763854]), 'rank_test_score': array([4, 2, 1, 2], dtype=int32)}

"""

model = KNeighborsClassifier(n_neighbors=7)
model.fit(x_train, y_train)
x = [[5.1, 3.5, 1.4, 0.2]]
x = pre.transform(x)
y_predict = model.predict(x_test)

print(accuracy_score(y_test, y_predict))

"""
0.9333333333333333
"""
