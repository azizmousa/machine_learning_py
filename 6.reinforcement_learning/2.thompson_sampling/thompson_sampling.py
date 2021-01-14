import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
dataset = pd.read_csv("dataset.csv")

rows, cols = dataset.shape

rows = 300

selected_ad = []
ad_reworded_1 = [0] * cols
ad_reworded_0 = [0] * cols

for n in range(0, rows):
    ad_index = 0
    max_ad_selected = 0
    for i in range(0, cols):
        random_beta = random.betavariate(ad_reworded_1[i] + 1, ad_reworded_0[i] + 1)
        if random_beta > max_ad_selected:
            max_ad_selected = random_beta
            ad_index = i

    if dataset.values[n, ad_index] == 0:
        ad_reworded_0[ad_index] += 1
    else:
        ad_reworded_1[ad_index] += 1
    selected_ad.append(ad_index)

plt.hist(selected_ad)
plt.show()
