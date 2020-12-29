import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv("Position_Salaries.csv")

X = dataset.iloc[:, 1].values
y = dataset.iloc[:, 2].values

plt.scatter(X, y, color="red")
plt.plot(X, y, color="blue")
plt.title("Position Salary Figure")
plt.xlabel("Salary")
plt.ylabel("Level")
plt.show()

