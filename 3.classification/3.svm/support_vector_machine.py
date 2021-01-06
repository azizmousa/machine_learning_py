import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score, plot_confusion_matrix

dataset = pd.read_csv("data.csv")
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
y = y.reshape(-1, 1)

x_scaler = StandardScaler()
X = x_scaler.fit_transform(X)

x_train, x_test, y_tarin, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

classifier = SVC(kernel="linear", random_state=0)
classifier.fit(x_train, y_tarin.flatten())
print(classifier.predict(x_scaler.fit_transform([[30, 87000]])))

y_hat = classifier.predict(x_test)

tn, fn, fp, tp = confusion_matrix(y_test, y_hat).ravel()
print(f"Linear confusion: tn: {tn}, fn: {fn}, fp: {fp}, tp: {tp}")

accuracy = accuracy_score(y_test, y_hat)
print("linear accuracy = ", accuracy)

plot_confusion_matrix(classifier, x_test, y_test)
plt.show()

non_linear_classifier = SVC(kernel='rbf', random_state=0)
non_linear_classifier.fit(x_train, y_tarin.flatten())

non_linear_y_hat = non_linear_classifier.predict(x_test)

nl_tn, nl_fn, nl_fp, nl_tp = confusion_matrix(y_test, non_linear_y_hat).ravel()
print(f"non linear confusion:  tn: {nl_tn}, fn: {nl_fn}, fp: {nl_fp}, tp: {nl_tp}")

accuracy = accuracy_score(y_test, non_linear_y_hat)
print("non linear accuracy = ", accuracy)

plot_confusion_matrix(non_linear_classifier, x_test, y_test)
plt.show()
