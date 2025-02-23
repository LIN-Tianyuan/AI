#  0.Import package
import os

os.environ["OMP_NUM_THREADS"] = "1"
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score

# 1.Build data sets
x, y = make_blobs(n_samples=1000, n_features=2, centers=[[-1, -1], [4, 4], [8, 8], [2, 2, ]],
                  cluster_std=[0.4, 0.2, 0.3, 0.2])

# 2.Iterate over different values of k to obtain sse
temp_list = []
for k in range(2, 100):
    model = KMeans(n_clusters=k)
    model.fit(x)
    y_pred = model.predict(x)
    temp_list.append(calinski_harabasz_score(x, y_pred))

# 3.Draw image
plt.figure()
plt.grid()
plt.plot(range(2, 100), temp_list, 'or-')
plt.show()
