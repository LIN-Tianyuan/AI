# 1.Toolkits
from sklearn.neighbors import KNeighborsClassifier

# 2.Data (feature engineering)
# Categorization
x = [[0,2,3],[1,3,4],[3,5,6],[4,7,8],[2,3,4]]
y = [0,0,1,1,0]

# 3.Instantiation
model = KNeighborsClassifier(n_neighbors=3)

# 4.Train
model.fit(x,y)

# 5.Predict
print(model.predict([[4,4,5]]))



