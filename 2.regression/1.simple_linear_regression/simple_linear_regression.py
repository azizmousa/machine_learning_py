import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
# load the dataset
dataset = pd.read_csv("Salary_Data.csv")

# devide the dataset to features and dependent values
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
plt.scatter(X, y, color="red")
plt.title("Salary, Experience dataset")
plt.xlabel("experience")
plt.ylabel("salary")
plt.show()

# handle the empty values in the num columns
sim = SimpleImputer(missing_values=np.NAN, strategy="mean")
X[:] = sim.fit_transform(X[:])
y = sim.fit_transform(y.reshape(-1, 1))

# split dataset to training_set, testing_set
Xtrain, Xtest, ytrain, ytest = train_test_split(X, y,
                                                test_size=0.2,
                                                random_state=1)
# load the regression model
regressor = LinearRegression()

# fit the model with the training data
regressor.fit(Xtrain, ytrain)

# test the model using predict method
y_pred = regressor.predict(Xtest)

# visualising the model data
plt.scatter(Xtrain, ytrain, color="red")
plt.plot(Xtrain, regressor.predict(Xtrain), color="blue")
plt.title("Salay VS Experience(Train set)")
plt.xlabel("Experience")
plt.ylabel("Salary")
plt.show()

plt.scatter(Xtest, ytest, color="red")
plt.plot(Xtrain, regressor.predict(Xtrain), color="blue")
plt.title("Salay VS Experience(Test set)")
plt.xlabel("Experience")
plt.ylabel("Salary")
plt.show()

print(regressor.predict([[12]]))
print(f"y = {regressor.intercept_[0]} + {regressor.coef_}*X")
