import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from apyori import apriori

dataset = pd.read_csv("market.csv", header=None)

# get the number of the rows and columns in the dataset
rows, cols = dataset.shape

# conver the dataframe to python list
transactions1 = []
for i in range(0, rows):
    transactions1.append(list(dataset.values[i, :].astype(str)))

# apply the apriori algorithm
rules = apriori(transactions1, min_support=.003, min_confidence=0.2, min_lift=3, min_length=2, max_length=2)
results = list(rules)
print(results)


# function to convert the results list to prepared list to use it in data frame
def construct(rules_results):
    lhs = [tuple(result[2][0][0])[0] for result in rules_results]
    rhs = [tuple(result[2][0][1])[0] for result in rules_results]
    support = [result[1] for result in rules_results]
    confidence = [result[2][0][2] for result in rules_results]
    lifts = [result[2][0][3] for result in rules_results]
    return list(zip(lhs, rhs, support, confidence, lifts))


# create dataframe from the prepared list
results_dataframe = pd.DataFrame(construct(results), columns=['Left side', 'right side',
                                                              'support', 'confidence', 'lift'])

# sort the dataframe by the larger lift
results_dataframe = results_dataframe.nlargest(n=10, columns='lift')
