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

plt.figure(2)
plt.scatter(X[y_clusters == 0, 0], X[y_clusters == 0, 1], color="cyan", label="cluster1")
plt.scatter(X[y_clusters == 1, 0], X[y_clusters == 1, 1], color="magenta", label="cluster2")
plt.scatter(X[y_clusters == 2, 0], X[y_clusters == 2, 1], color="blue", label="cluster3")
plt.scatter(X[y_clusters == 3, 0], X[y_clusters == 3, 1], color="yellow", label="cluster4")
plt.scatter(X[y_clusters == 4, 0], X[y_clusters == 4, 1], color="green", label="cluster5")
plt.legend()
plt.xlabel("annual income")
plt.ylabel("spending score")
plt.show()
