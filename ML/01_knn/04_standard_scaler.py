# 1.Toolkits
from sklearn.preprocessing import StandardScaler

# 2.Data (only features)
x = [[90, 2, 10, 40], [60, 4, 15, 45], [75, 3, 13, 46]]

# 3.Instantiation(standardization)
process = StandardScaler()

# 4.fit and transform
data = process.fit_transform(x)

print(process.mean_)
print(process.var_)