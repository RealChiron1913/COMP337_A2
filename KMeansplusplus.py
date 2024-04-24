#  Implement k-means++ clustering algorithm and cluster the dataset provided using it. Vary the value
# of k from 1 to 9 and compute the Silhouette coefficient for each set of clusters. Plot k in the horizontal
# axis and the Silhouette coefficient in the vertical axis in the same plot


import numpy as np
import matplotlib.pyplot as plt

def load_data(fname):
    """Load data from a file into a NumPy array."""
    features = []
    with open(fname) as F:
        for line in F:
            parts = line.strip().split()
            features.append(np.array(parts[1:], dtype=float))
    return np.array(features)

def ComputeDistance(x, y):
    """Calculate the Euclidean distance between two points."""
    return np.linalg.norm(x - y)

def initialSelection(dataset, k):
    """Select initial centroids using the K-means++ algorithm."""
    np.random.seed(42)  # Ensure reproducibility
    initial_index = np.random.choice(len(dataset))
    centroids = [dataset[initial_index]]
    
    for _ in range(1, k):
        distances = np.array([min([ComputeDistance(x, centroid) for centroid in centroids]) for x in dataset])
        probabilities = distances**2 / np.sum(distances**2)
        next_index = np.random.choice(len(dataset), p=probabilities)
        centroids.append(dataset[next_index])
    
    return np.array(centroids)

def assignClusterIds(dataset, centers):
    """Assign each data point to the nearest cluster by centroid."""
    cluster_ids = [np.argmin([ComputeDistance(point, center) for center in centers]) for point in dataset]
    return np.array(cluster_ids)

def computeClusterRepresentatives(dataset, cluster_ids, k):
    """Compute new centroids as the mean of the data points in each cluster."""
    return np.array([dataset[cluster_ids == i].mean(axis=0) for i in range(k)])

def KMeansplusplus(dataset, maxIter=100, max_k=9):
    """Run the K-means algorithm, compute silhouette scores, and plot them."""
    silhouette_scores = [0]  # Silhouette score for k=1 is not defined
    for k in range(2, max_k+1):
        centers = initialSelection(dataset, k)
        for _ in range(maxIter):
            cluster_ids = assignClusterIds(dataset, centers)
            new_centers = computeClusterRepresentatives(dataset, cluster_ids, k)
            if np.allclose(centers, new_centers):
                break
            centers = new_centers
        clusters = [np.where(cluster_ids == i)[0] for i in range(k)]
        score = silhouette(dataset, clusters, cluster_ids, distanceMatrix(dataset))
        silhouette_scores.append(score)
    plot_silhouette(silhouette_scores)

def distanceMatrix(dataset):
    """Create a matrix of all pairwise distances between data points in the dataset."""
    N = len(dataset)
    distMatrix = np.zeros((N, N))
    for i in range(N):
        for j in range(i + 1, N):
            distMatrix[i][j] = distMatrix[j][i] = ComputeDistance(dataset[i], dataset[j])
    return distMatrix

def silhouette(dataset, clusters, cluster_ids, distMatrix):
    """Calculate the silhouette score for the clustering."""
    silhouette_values = []
    for i in range(len(dataset)):
        a = np.mean([distMatrix[i][j] for j in clusters[cluster_ids[i]] if i != j])
        b = min(np.mean(distMatrix[i][clusters[j]]) for j in range(len(clusters)) if j != cluster_ids[i])
        silhouette_values.append((b - a) / max(a, b))
    return np.mean(silhouette_values)

def plot_silhouette(silhouette_scores):
    """Plot silhouette scores against the number of clusters."""
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, 10), silhouette_scores, marker='o')
    plt.title('Silhouette Coefficient for Various Numbers of Clusters')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Silhouette Coefficient')
    plt.grid(True)
    plt.show()

if __name__ == '__main__':
    dataset = load_data('dataset')
    KMeansplusplus(dataset)
