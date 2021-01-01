import numpy as np
import pandas as pd
from linear_regression_model import LinearRegressionModel

dataset = pd.read_csv("Position_Salaries.csv")
X = dataset.iloc[:, 1].values
y = dataset.iloc[:, 2].values

model = LinearRegressionModel(x_train=X, y_train=y)
