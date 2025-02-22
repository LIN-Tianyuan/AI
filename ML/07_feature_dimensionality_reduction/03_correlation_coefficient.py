from sklearn.datasets import load_iris
from scipy.stats import pearsonr, spearmanr

# Correlation coefficient
x, y = load_iris(return_X_y=True)
x1 = x[:, 2]
x2 = x[:, 1]

print(pearsonr(x1, x2))
print(spearmanr(x1, x2))
