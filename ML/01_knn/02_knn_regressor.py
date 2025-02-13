# 1.Toolkits
from sklearn.neighbors import KNeighborsRegressor

# 2.Data (feature engineering)
# Categorization
x = [[0,1,2],[1,2,3],[2,3,4],[3,4,5]]
y = [0.1,0.2,0.3,0.4]

# 3.Instantiation
model = KNeighborsRegressor(n_neighbors=3)

# 4.Train
model.fit(x,y)

# 5.Predict
print(model.predict([[4,4,5]]))
