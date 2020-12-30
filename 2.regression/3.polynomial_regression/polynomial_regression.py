import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
dataset = pd.read_csv("Position_Salaries.csv")

X = dataset.iloc[:, 1].values.reshape(-1, 1)
y = dataset.iloc[:, 2].values.reshape(-1, 1)


def plot(x_scatter, y_scatter, x_p, y_p, title, xlabel, ylabel, plot_color):
    plt.title(title)
    plt.scatter(x_scatter, y_scatter, color="red")
    plt.plot(x_p, y_p, color=plot_color)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()


plot(X, y, X, y, "Position Salary Figure", "Level", "Salary", "green")
poly = PolynomialFeatures(degree=4)
X_poly = poly.fit_transform(X)

regressor = LinearRegression()
regressor.fit(X_poly, y)
y_pred = regressor.predict(X_poly)
plot(X, y, X, y_pred, "predict Figure", "Level", "Salary", "blue")
