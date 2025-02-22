from sklearn.feature_selection import VarianceThreshold
import pandas as pd

# Low variance filtration
data = pd.read_csv('垃圾邮件分类数据.csv')
print(data.shape)

transform = VarianceThreshold(threshold=0.1)
x = transform.fit_transform(data)
print(x.shape)

"""
(971, 25734)
(971, 1044)
"""