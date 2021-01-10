import pandas as pd
from matplotlib import pyplot as plt
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as sch
dataset = pd.read_csv("mall.csv")
X = dataset.iloc[:, [3, 4]].values

dendogram = sch.dendrogram(sch.linkage(X, method='ward'))
plt.title("dendogram")
plt.xlabel("data points")
plt.ylabel("euclidean distance")
plt.show()

# accourding to the dendogram the best classes number if 3 or 5

model = AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='ward')
y_clusters = model.fit_predict(X)
print(y_clusters)
