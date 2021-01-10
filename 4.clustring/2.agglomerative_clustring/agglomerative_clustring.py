import pandas as pd
from matplotlib import pyplot as plt
dataset = pd.read_csv("mall.csv")
X = dataset.iloc[:, [3, 4]].values


