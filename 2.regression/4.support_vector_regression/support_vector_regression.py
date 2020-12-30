import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

dataset = pd.read_csv("Position_Salaries.csv")
X = dataset.iloc[:, 1].values
y = dataset.iloc[:, 2].values
X = X.reshape(-1, 1)
y.reshape(-1, 1)


# plot function to draw data figures
def plot(x_scat, y_scat, x_plt=None, y_plt=None, title="Figure", xlabel="X axis",
         ylabel="Y axis", plot_color="blue", scat_color="red", draw_plot=True, figure=1):
    plt.figure(figure)
    plt.title(title)
    plt.scatter(x_scat, y_scat, color=scat_color)
    if draw_plot:
        plt.plot(x_plt, y_plt, color=plot_color)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.show()


plot(X, y, X, y, xlabel="Level", ylabel="Salary", draw_plot=True, figure=1)

