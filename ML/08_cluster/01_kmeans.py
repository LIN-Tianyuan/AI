# 0.Import package
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import calinski_harabasz_score, silhouette_score

# 1.Build data sets
x, y = make_blobs(n_samples=1000, n_features=2, centers=[[-1, -1], [4, 4], [8, 8], [2, 2, ]],
                  cluster_std=[0.4, 0.2, 0.3, 0.2])
# plt.figure()
# plt.scatter(x[:, 0], x[:, 1])
# plt.show()

# 2.Model Training Predictions
model = KMeans(n_clusters=4)
y_prd = model.fit_predict(x)

# plt.figure()
# plt.scatter(x[:, 0], x[:, 1], c=y_prd)
# plt.show()



# 3.Evaluation
print(silhouette_score(x,y_prd))
