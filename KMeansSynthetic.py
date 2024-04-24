# Generate synthetic data of same size (i.e. same number of data points) as the dataset provided and use (10)
# this data to cluster K Means. Plot k in the horizontal axis and the Silhouette coefficient in the vertical
# axis in the same plot.

import numpy as np

def generateSyntheticData(dataset):
    """Generate synthetic data of the same size as the dataset provided.

    Args:
        dataset (str): Filename of the dataset.

    Returns:
        np.array: Synthetic data of the same size as the dataset.
    """
    np.random.seed(42)
    return np.random.rand(*get_shape(dataset))

def get_shape(filename):
    """get the shape of the provided dataset.

    Args:
        filename (str): Filename of the dataset.

    Returns:
        tuple: Shape of the dataset.
    """
    data = []

    with open(filename) as f:
        for line in f:
            parts = line.split()
            data.append(np.array(parts[1:], dtype=float))

    return np.array(data).shape

def ComputeDistance(x, y):
    """Calculate the Euclidean distance between two points.
    
    Args:
        x (np.array): First point.
        y (np.array): Second point.
        
    Returns:
        float: Euclidean distance between x and y.
    """
    return np.linalg.norm(x - y)

def initialSelection(dataset, k):
    """Select initial centroids randomly from the dataset.
    
    Args:
        dataset (np.array): The dataset from which centroids are to be selected.
        k (int): Number of clusters.
        
    Returns:
        np.array: Initial centroids selected randomly from the dataset.
    """
    np.random.seed(42)  # Ensure reproducibility
    indices = np.random.choice(dataset.shape[0], k, replace=False)
    return dataset[indices]

def assignClusterIds(dataset, centers):
    """Assign each data point to the nearest cluster by centroid.
    
    Args:
        dataset (np.array): The dataset.
        centers (np.array): Current centroids.
        
    Returns:
        np.array: Cluster IDs for each data point.
    """
    cluster_ids = [np.argmin([ComputeDistance(point, center) for center in centers]) for point in dataset]
    return np.array(cluster_ids)

def computeClusterRepresentatives(dataset, cluster_ids, k):
    """Compute new centroids as the mean of the data points in each cluster.
    
    Args:
        dataset (np.array): The dataset.
        cluster_ids (np.array): Array of cluster IDs for each data point.
        k (int): Number of clusters.
        
    Returns:
        np.array: New centroids.
    """
    new_centers = []
    for i in range(k):
        cluster_points = dataset[cluster_ids == i]
        new_centers.append(np.mean(cluster_points, axis=0))
    return np.array(new_centers)

def silhouette(dataset, cluster_ids, k):
    """Compute the Silhouette coefficient for the clustering.
    
    Args:
        dataset (np.array): The dataset.
        cluster_ids (np.array): Array of cluster IDs for each data point.
        k (int): Number of clusters.
        
    Returns:
        float: Silhouette coefficient.
    """
    silhouette_values = []
    for i in range(k):
        cluster_points = dataset[cluster_ids == i]
        for point in cluster_points:
            a = np.mean([ComputeDistance(point, other) for other in cluster_points])
            b = min([np.mean([ComputeDistance(point, other) for other in dataset[cluster_ids == j]]) for j in range(k) if j != i])
            silhouette_values.append((b - a) / max(a, b))
    return np.mean(silhouette_values)

def KMeansSynthetic(dataset, maxIter=100, max_k=9):
    """Run the K-means algorithm, compute silhouette scores, and plot them.

    Args:
        dataset (np.array): The dataset on which k-means is to be run.
        maxIter (int): Maximum number of iterations for convergence.
    """
    silhouette_scores = [0]  # Silhouette score for k=1 is not defined
    for k in range(2, max_k+1):
        centers = initialSelection(dataset, k)
        for _ in range(maxIter):
            cluster_ids = assignClusterIds(dataset, centers)
            new_centers = computeClusterRepresentatives(dataset, cluster_ids, k)
            if np.allclose(centers, new_centers):
                break
            centers = new_centers
        score = silhouette(dataset, cluster_ids, k)
        silhouette_scores.append(score)

    plot_silhouette(silhouette_scores)

def plot_silhouette(silhouette_scores):
    """Plot the silhouette scores for different values of k.

    Args:
        silhouette_scores (list): List of silhouette scores.
    """
    import matplotlib.pyplot as plt
    plt.plot(range(1, 10), silhouette_scores, marker='o')
    plt.xlabel('Number of clusters (k)')
    plt.ylabel('Silhouette coefficient')
    plt.title('Silhouette coefficient vs. Number of clusters')
    plt.show()

def distanceMatrix(dataset):
    """Create a matrix of all pairwise distances between data points in the dataset.
    
    Args:
        dataset (np.array): Dataset of data points.
        
    Returns:
        np.array: Matrix of distances.
    """
    N = len(dataset)
    distMatrix = np.zeros((N, N))
    for i in range(N):
        for j in range(i + 1, N):
            distMatrix[i][j] = distMatrix[j][i] = ComputeDistance(dataset[i], dataset[j])
    return distMatrix


if __name__ == '__main__':
    dataset = 'dataset'
    synthetic_data = generateSyntheticData(dataset)
    KMeansSynthetic(synthetic_data)
