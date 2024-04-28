import numpy as np
import matplotlib.pyplot as plt

def load_data(fname):
    """Load data from a file into a NumPy array.
    
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
        np.array: Array of new centroids.
    """
    return np.array([dataset[cluster_ids == i].mean(axis=0) for i in range(k)])


def KMeans(dataset, maxIter=100, max_k=9):
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
        clusters = [np.where(cluster_ids == i)[0] for i in range(k)]
        score = silhouette(dataset, clusters, cluster_ids, distanceMatrix(dataset))
        silhouette_scores.append(score)
    plot_silhouette(silhouette_scores)


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

def silhouette(dataset, clusters, cluster_ids, distMatrix):
    """Calculate the silhouette score for the clustering.

    Args:
        dataset (np.array): The dataset.
        clusters (list of arrays): List of clusters where each cluster contains indices of data points.
        cluster_ids (np.array): Cluster IDs for each data point.
        distMatrix (np.array): Precomputed distance matrix.

    Returns:
        float: Mean silhouette score.
    """
    silhouette_values = []
    for i in range(len(dataset)):
        a = np.mean([distMatrix[i][j] for j in clusters[cluster_ids[i]] if i != j])
        b = min(np.mean(distMatrix[i][clusters[j]]) for j in range(len(clusters)) if j != cluster_ids[i])
        silhouette_values.append((b - a) / max(a, b))
    return np.mean(silhouette_values)


def plot_silhouette(silhouette_scores):
    """Plot silhouette scores against the number of clusters.
    
    Args:
        silhouette_scores (list): List of silhouette scores for k from 1 to 9.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, 10), silhouette_scores, marker='o')
    plt.title('Silhouette Coefficient for Various Numbers of Clusters')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Silhouette Coefficient')
    plt.grid(True)
    plt.savefig('KMeans.png')

if __name__ == '__main__':
    dataset = load_data('dataset')
    KMeans(dataset)
