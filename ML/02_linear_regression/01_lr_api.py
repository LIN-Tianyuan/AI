from sklearn.linear_model import LinearRegression

# 1.Data
x = [[160], [166], [172], [174], [180]]
y = [56.3, 60.6, 65.1, 68.5, 75]
# 2.Model training
# 2.1 Instantiation
model = LinearRegression()
# 2.2 Train
model.fit(x, y)

# (weight)/(bias)
print(model.coef_)
print(model.intercept_)

# 3.Model prediction
print(model.predict([[176]]))
