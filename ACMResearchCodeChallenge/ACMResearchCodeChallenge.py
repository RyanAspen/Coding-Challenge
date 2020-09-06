from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score
import pandas as pd

# Get Data from Spreadsheet
clusterDF = pd.read_csv("ClusterPlot.csv")
clusterUsefulColumns = clusterDF[['V1','V2']]

# https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html#sklearn.cluster.KMeans.fit
# https://www.datanovia.com/en/lessons/determining-the-optimal-number-of-clusters-3-must-know-methods/
# https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html
# I predict that there are at least 2 clusters and no more than 5 clusters

silhouette_scores = []

possible_cluster_numbers = [2,3,4,5,6,7,8,9,10]

for k in possible_cluster_numbers:
    kmeans = KMeans(n_clusters=k)
    cluster_labels = kmeans.fit_predict(clusterUsefulColumns)
    silhouette_avg = silhouette_score(clusterUsefulColumns, cluster_labels)
    silhouette_scores.append(silhouette_avg)

best_silhouette_score_index = silhouette_scores.index(max(silhouette_scores))
best_cluster_number = possible_cluster_numbers[best_silhouette_score_index]
print("There are most likely", best_cluster_number, "clusters in the dataset.")