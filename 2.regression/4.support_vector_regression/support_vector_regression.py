import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
dataset = pd.read_csv("Position_Salaries.csv")
X = dataset.iloc[:, 1].values
y = dataset.iloc[:, 2].values
X = X.reshape(-1, 1)
y = y.reshape(-1, 1)


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


plot(X, y, X, y, xlabel="Level", ylabel="Salary", plot_color="green", draw_plot=True, figure=1)

X_ss = StandardScaler()
y_ss = StandardScaler()
X_scaled = X_ss.fit_transform(X)
y_scaled = y_ss.fit_transform(y)

svr_model = SVR(kernel="rbf")
svr_model.fit(X_scaled, y_scaled.flatten())
prediction = svr_model.predict(X_scaled)
prediction = y_ss.inverse_transform(prediction)

plot(X, y, X, prediction, xlabel="Level", ylabel="salary", plot_color="blue", draw_plot=True, figure=1)
