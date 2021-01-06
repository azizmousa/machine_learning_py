import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

dataset = pd.read_csv("data.csv")
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
y = y.reshape(-1, 1)

x_scaler = StandardScaler()
X = x_scaler.fit_transform(X)

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
