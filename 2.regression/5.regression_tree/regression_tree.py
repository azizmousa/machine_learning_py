import numpy as np
import pandas as pd
from Ploter import plot
from sklearn import tree
data_path = "Position_Salaries.csv"
dataset = pd.read_csv(data_path)
X = dataset.iloc[:, 1].values
y = dataset.iloc[:, 2].values
X = X.reshape(-1, 1)
y = y.reshape(-1, 1)

plot(X, y, X, y, xlabel="Level", ylabel="Salary", title="Salary Level", plot_color="green", figure=1)

reg_tree_model = tree.DecisionTreeRegressor(random_state=0)
reg_tree_model.fit(X, y)
tree.plot_tree(reg_tree_model)
# predict single value 4.5 >>>>>>>> 80000
print(reg_tree_model.predict([[6.5]]))
X_arn = np.arange(min(X), max(X), 0.1)
X_arn = X_arn.reshape(-1, 1)
plot(X, y, X_arn, reg_tree_model.predict(X_arn), xlabel="Level", ylabel="Salary",
     title="Salary Level", plot_color="blue", figure=2)
