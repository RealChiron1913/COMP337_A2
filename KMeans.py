# Implement k-means clustering algorithm and cluster the dataset provided using it. 
# Vary the value of k from 1 to 9 and compute the Silhouette coefficient for each set of clusters. 
# Plot k in the horizontal axis and the Silhouette coefficient in the vertical axis in the same plot.
import numpy as np

from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# Already implemented
def load_data(fname):
    features = []
    
    with open(fname) as F:
        for line in F:
            p = line.strip().split(' ')
            features.append(np.array(p[1:], float))
    return np.array(features)


def ComputeDistance(x, y): 
    return np.linalg.norm(x - y)

# Implementing the remaining functions
def initialSelection(dataset, k):
    np.random.seed(42)  # Ensure reproducibility
    indices = np.random.choice(dataset.shape[0], k, replace=False)
    return dataset[indices]

def assignClusterIds(dataset, centers):
    cluster_ids = []
    for point in dataset:
        distances = [ComputeDistance(point, center) for center in centers]
        cluster_id = np.argmin(distances)
        cluster_ids.append(cluster_id)
    return np.array(cluster_ids)

def computeClusterRepresentatives(dataset, cluster_ids, k):
    new_centers = []
    for i in range(k):
        points_in_cluster = dataset[cluster_ids == i]
        new_center = points_in_cluster.mean(axis=0)
        new_centers.append(new_center)
    return np.array(new_centers)

def kmeans(dataset, k):
    centers = initialSelection(dataset, k)
    for _ in range(100):  # Max iterations
        cluster_ids = assignClusterIds(dataset, centers)
        new_centers = computeClusterRepresentatives(dataset, cluster_ids, k)
        if np.allclose(centers, new_centers):
            break
        centers = new_centers
    return cluster_ids

# Plot silhouette scores
def plot_silhouette(dataset):
    silhouette_scores = []
    k_values = range(1, 10)
    for k in k_values:
        if k == 1:
            silhouette_scores.append(-1)  # Silhouette score is not meaningful for k=1.
        else:
            cluster_ids = kmeans(dataset, k)
            score = silhouette_score(dataset, cluster_ids)
            silhouette_scores.append(score)
    
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, silhouette_scores, marker='o')
    plt.title('Silhouette Coefficient for Various Numbers of Clusters')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Silhouette Coefficient')
    plt.grid(True)
    plt.show()

# For demonstration, let's create a random dataset
dataset = load_data('dataset')
# Plotting silhouette scores for the demonstration dataset
plot_silhouette(dataset)


