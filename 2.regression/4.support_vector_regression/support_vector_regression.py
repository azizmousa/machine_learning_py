import numpy as np
import pandas as pd
from Ploter import plot
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
dataset = pd.read_csv("Position_Salaries.csv")
X = dataset.iloc[:, 1].values
y = dataset.iloc[:, 2].values
X = X.reshape(-1, 1)
y = y.reshape(-1, 1)

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
