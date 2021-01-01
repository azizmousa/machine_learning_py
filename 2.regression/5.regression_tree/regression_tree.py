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

reg_tree_model = tree.DecisionTreeRegressor()
reg_tree_model.fit(X, y)
tree.plot_tree(reg_tree_model)
# predict single value 4.5 >>>>>>>> 80000
print(reg_tree_model.predict([[4.5]]))
