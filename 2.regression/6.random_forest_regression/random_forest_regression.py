import numpy as np
import pandas as pd
from Ploter import plot
from sklearn.ensemble import RandomForestRegressor

data_path = "Position_Salaries.csv"
dataset = pd.read_csv(data_path)
X = dataset.iloc[:, 1].values
y = dataset.iloc[:, 2].values

X = X.reshape(-1, 1)
y = y.reshape(-1, 1)

plot(X, y, X, y, xlabel="Level", ylabel="Salary", title="Salary Level", plot_color="green", figure=1)

rfr_model = RandomForestRegressor(n_estimators=10, random_state=0)
rfr_model.fit(X, y.flatten())

X_arn = np.arange(min(X), max(X), 0.1)
X_arn = X_arn.reshape(-1, 1)

plot(X, y, X_arn, rfr_model.predict(X_arn), xlabel="Level", ylabel="Salary",
     title="Salary Level", plot_color="blue", figure=2)

# predict single value 4.5 >>>>>>> 89000
print(rfr_model.predict([[4.5]]))
