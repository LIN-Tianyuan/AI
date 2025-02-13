# 1.Toolkits
from sklearn.preprocessing import MinMaxScaler

# 2.Data (only features)
x = [[90, 2, 10, 40], [60, 4, 15, 45], [75, 3, 13, 46]]

# 3.Instantiation(normalization)
process = MinMaxScaler()

# 4.fit and transform
data = process.fit_transform(x)

print(data)