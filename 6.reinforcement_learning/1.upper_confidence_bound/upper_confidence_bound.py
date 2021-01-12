import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math

dataset = pd.read_csv("dataset.csv")
ad_selected = []
num_rounds = 10000
num_ads = 10
num_of_selection = [0] * num_ads
sum_of_rewards = [0] * num_ads

for n in range(0, num_rounds):
    max_ucb = 0
    mx_ad_index = 0
    for i in range(0, num_ads):
        if num_of_selection[i] > 0:
            avg_reward = sum_of_rewards[i] / num_of_selection[i]
            delta_i = math.sqrt((3/2) * (math.log(n+1) / num_of_selection[i]))
            ucb = avg_reward + delta_i
        else:
            ucb = 1e400
        if ucb > max_ucb:
            max_ucb = ucb
            mx_ad_index = i
    ad_selected.append(mx_ad_index)
    num_of_selection[mx_ad_index] += 1
    reward = dataset.values[n, mx_ad_index]
    sum_of_rewards[mx_ad_index] += reward

plt.hist(ad_selected)
plt.show()

