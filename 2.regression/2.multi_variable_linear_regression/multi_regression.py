import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# load dataset
dataset = pd.read_csv("50_Startups.csv")
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# handle missing data
imp = SimpleImputer(missing_values=np.nan, strategy="mean")
X[:, 0:3] = imp.fit_transform(X[:, 0:3])
y = imp.fit_transform(y.reshape(-1, 1))

# convert the catigorical data
ct = ColumnTransformer(transformers=[("encoder",
                                     OneHotEncoder(),
                                     [3])], remainder="passthrough")
X = np.array(ct.fit_transform(X))

# split dataset to train, test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=1)

# training the linear regression model
# the LinearRegression class take care of
# the dummy variable and
# the selecting the best features with the highest p-value
regressor = LinearRegression()
regressor.fit(X_train, y_train)

