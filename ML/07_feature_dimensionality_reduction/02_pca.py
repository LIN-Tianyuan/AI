from sklearn.decomposition import PCA
from sklearn.datasets import load_iris


# pca
x, y = load_iris(return_X_y=True)
print(x)
pca1 = PCA(n_components=0.95)
print(pca1.fit_transform(x))
pca1 = PCA(n_components=3)
print(pca1.fit_transform(x))
