import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
dataset = pd.read_csv("mall.csv")
X = dataset.iloc[:, 3:].values

# ct = ColumnTransformer([("encoder", OneHotEncoder(), [0])], remainder="passthrough")
# X = ct.fit_transform(X)

wcss = []
for i in range(1, 11):
    model = KMeans(n_clusters=i, init='k-means++')
    model.fit(X)
    wcss.append(model.inertia_)

plt.figure(1)
plt.plot(range(1, 11), wcss)
plt.title("the elbow method")
plt.show()
# from the elbow graph the best k clusters is 5

kmeans = KMeans(n_clusters=5, init='k-means++')
y_kmean = kmeans.fit_predict(X)
print(y_kmean)

plt.figure(2)
plt.scatter(X[y_kmean == 0, 0], X[y_kmean == 0, 1], color="cyan", label="cluster1")
plt.scatter(X[y_kmean == 1, 0], X[y_kmean == 1, 1], color="magenta", label="cluster2")
plt.scatter(X[y_kmean == 2, 0], X[y_kmean == 2, 1], color="blue", label="cluster3")
plt.scatter(X[y_kmean == 3, 0], X[y_kmean == 3, 1], color="yellow", label="cluster4")
plt.scatter(X[y_kmean == 4, 0], X[y_kmean == 4, 1], color="green", label="cluster5")
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], color="red", label="centroids")
plt.legend()
plt.xlabel("annual income")
plt.ylabel("spending score")
plt.show()
