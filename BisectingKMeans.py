import numpy as np
import matplotlib.pyplot as plt

def load_data(fname):
    """
    Load data from a file into a NumPy array.

    Args:
        fname (str): Filename of the data file.

    Returns:
        np.array: Array of data points where each row represents a data point.
    """
    features = []
    with open(fname) as F:
        for line in F:
            parts = line.strip().split()
            features.append(np.array(parts[1:], dtype=float))
    return np.array(features)

def ComputeDistance(x, y):
    """
    Calculate the Euclidean distance between two points.

    Args:
        x (np.array): First point.
        y (np.array): Second point.

    Returns:
        float: Euclidean distance between x and y.

    """
    return np.linalg.norm(x - y)

def initialSelection(dataset, k):
    """
    Select initial centroids randomly from the dataset.

    Args:
        dataset (np.array): The dataset from which centroids are to be selected.
        k (int): Number of clusters.

    Returns:
        np.array: Initial centroids selected randomly from the dataset.

    """
    np.random.seed(42)
    indices = np.random.choice(dataset.shape[0], k, replace=False)
    return dataset[indices]

def assignClusterIds(dataset, centers):
    """
    Assign each data point to the nearest cluster by centroid.

    Args:
        dataset (np.array): The dataset.
        centers (np.array): Current centroids.

    Returns:
        np.array: Cluster IDs for each data point.
    """
    cluster_ids = [np.argmin([ComputeDistance(point, center) for center in centers]) for point in dataset]
    return np.array(cluster_ids)

def computeClusterRepresentatives(dataset, cluster_ids, k):
    """
    Compute new centroids as the mean of the data points in each cluster.

    Args:
        dataset (np.array): The dataset.
        cluster_ids (np.array): Array of cluster IDs for each data point.
        k (int): Number of clusters.

    Returns:
        np.array: Array of new centroids.
    """
    return np.array([dataset[cluster_ids == i].mean(axis=0) for i in range(k)])

def bisectingKMeans(dataset, max_k, maxIter=100):
    """
    Run the bisecting K-means algorithm, compute silhouette scores, and plot them.

    Args:
        dataset (np.array): The dataset on which k-means is to be run.
        max_k (int): Maximum number of clusters to try.
        maxIter (int): Maximum number of iterations for K-means.

    Returns:
        float: Silhouette score for the best number of clusters.

    """
    clusters = [np.arange(len(dataset))]  # Start with one cluster containing all indices
    silhouette_scores = [0]  # Silhouette score for k=1 is not defined

    while len(clusters) < max_k:
        # Choose a cluster to split (largest one)
        largest_cluster_idx = np.argmax([len(cluster) for cluster in clusters])
        to_split = dataset[clusters[largest_cluster_idx]]

        # Perform standard K-means on the cluster to be split
        centers = initialSelection(to_split, 2)
        for _ in range(maxIter):
            cluster_ids = assignClusterIds(to_split, centers)
            new_centers = computeClusterRepresentatives(to_split, cluster_ids, 2)
            if np.allclose(centers, new_centers):
                break
            centers = new_centers

        # Replace the original cluster with the two new ones
        new_cluster1 = clusters[largest_cluster_idx][cluster_ids == 0]
        new_cluster2 = clusters[largest_cluster_idx][cluster_ids == 1]
        clusters[largest_cluster_idx] = new_cluster1
        clusters.append(new_cluster2)

        # Compute silhouette scores for current clustering
        cluster_ids_full = np.zeros(len(dataset), dtype=int)
        for idx, cluster in enumerate(clusters):
            cluster_ids_full[cluster] = idx
        distMatrix = distanceMatrix(dataset)
        score = silhouette(dataset, clusters, cluster_ids_full, distMatrix)
        silhouette_scores.append(score)

    plot_silhouette(silhouette_scores)

def distanceMatrix(dataset):
    """
    Create a matrix of all pairwise distances between data points in the dataset.

    Args:
        dataset (np.array): Dataset of data points.

    Returns:
        np.array: Matrix of distances.

    """
    N = len(dataset)
    distMatrix = np.zeros((N, N))
    for i in range(N):
        for j in range(i+1, N):
            distMatrix[i, j] = ComputeDistance(dataset[i], dataset[j])
            distMatrix[j, i] = distMatrix[i, j]
    return distMatrix

def silhouette(dataset, clusters, cluster_ids, distMatrix):
    """
    Calculate the silhouette score for the clustering.

    Args:
        dataset (np.array): The dataset.
        clusters (list): List of clusters where each cluster is a list of indices.
        cluster_ids (np.array): Array of cluster IDs for each data point.
        distMatrix (np.array): Matrix of pairwise distances between data points.

    Returns:
        float: Silhouette score for the clustering.

    """

    silhouette_vals = []
    for idx, point in enumerate(dataset):
        cluster_id = cluster_ids[idx]
        a = np.mean([distMatrix[idx, other] for other in clusters[cluster_id]])
        b = min([np.mean([distMatrix[idx, other] for other in cluster]) for i, cluster in enumerate(clusters) if i != cluster_id])
        silhouette_vals.append((b - a) / max(a, b))
    return np.mean(silhouette_vals)

def plot_silhouette(silhouette_scores):
    """
    Plot silhouette scores against the number of clusters.

    Args:
        silhouette_scores (list): List of silhouette scores for each number of clusters.

    """
    plt.plot(range(1, 10), silhouette_scores, marker='o')
    plt.title('Silhouette scores for bisecting K-means')
    plt.xlabel('Number of clusters')
    plt.ylabel('Silhouette score')
    plt.grid(True)
    # save the plot
    plt.savefig('BisectingKMeans.png')

if __name__ == '__main__':
    dataset = load_data('dataset')
    score = bisectingKMeans(dataset, 9)

