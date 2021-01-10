import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder

dataset = pd.read_csv("mall.csv")
X = dataset.iloc[:, 3:].values

# ct = ColumnTransformer([("encoder", OneHotEncoder(), [0])], remainder="passthrough")
# X = ct.fit_transform(X)


