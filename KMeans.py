# Implement k-means clustering algorithm and cluster the dataset provided using it. 
# Vary the value of k from 1 to 9 and compute the Silhouette coefficient for each set of clusters. 
# Plot k in the horizontal axis and the Silhouette coefficient in the vertical axis in the same plot.
import numpy as np

import matplotlib.pyplot as plt
import math


def load_data(fname):
    features = []
    
    with open(fname) as F:
        for line in F:
            p = line.strip().split(' ')
            features.append(np.array(p[1:], float))
    return np.array(features)


def ComputeDistance(x, y): 
    return np.linalg.norm(x - y) # Euclidean distance

def initialSelection(dataset, k):
    np.random.seed(22)  # Ensure reproducibility
    indices = np.random.choice(dataset.shape[0], k, replace=False)
    return dataset[indices]

def assignClusterIds(dataset, centers): # Assign each point to the closest cluster
    cluster_ids = []
    for point in dataset:
        distances = [ComputeDistance(point, center) for center in centers]
        cluster_id = np.argmin(distances)
        cluster_ids.append(cluster_id)
    return np.array(cluster_ids)

def computeClusterRepresentatives(dataset, cluster_ids, k): # Compute the new centers of the clusters
    new_centers = []
    for i in range(k):
        points_in_cluster = dataset[cluster_ids == i]
        new_center = points_in_cluster.mean(axis=0)
        new_centers.append(new_center)
    return np.array(new_centers)

def KMeans(dataset, maxIter=100):
    silhouette_scores = []
    silhouette_scores.append(0)  # Silhouette score for k=1 is not defined

    for k in range(2, 10):
        centers = initialSelection(dataset, k)
        iteration = 0
        # iterate until convergence
        while True:
            iteration += 1
            cluster_ids = assignClusterIds(dataset, centers)
            new_centers = computeClusterRepresentatives(dataset, cluster_ids, k)
            if np.array_equal(centers, new_centers) or iteration > maxIter:
                break
            centers = new_centers

        # Compute Silhouette score
        distMatrix = distanceMatrix(dataset)
        score = silhouette(dataset, [np.where(cluster_ids == i)[0] for i in range(k)], distMatrix)
        silhouette_scores.append(score)

    plot_silhouette(silhouette_scores)



# Plot silhouette scores
def plot_silhouette(silhouette_scores):

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, 10), silhouette_scores, marker='o')
    plt.title('Silhouette Coefficient for Various Numbers of Clusters')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Silhouette Coefficient')
    plt.grid(True)
    plt.show()
    

# Computes distances matrix for a given dataset and a disntace function 
def distanceMatrix(dataset, dist=ComputeDistance):
    # Compute the number of objects in the dataset
    N = len(dataset)
    
    # Distance matrix
    distMatrix = np.zeros((N, N))
    # Compute pairwise distances between the objects
    for i in range(N):
        for j in range (N):
            # Distance is symmetric, so compute the distances between i and j only once
            if i < j:
                distMatrix[i][j] = dist(dataset[i], dataset[j])
                distMatrix[j][i] = distMatrix[i][j]
            
    return distMatrix

# Computes Silhouette Coefficient for every object in the dataset with respect to the given clustering and distance matrix
def silhouetteCoefficient(dataset, clusters, distMatrix):
    # Compute the number of objects in the dataset
    N = len(dataset)
    
    silhouette = [0 for i in range(N)]
    a = [0 for i in range(N)]
    b = [math.inf for i in range(N)]
    
    for (i, obj) in enumerate(dataset):
        for (cluster_id, cluster) in enumerate(clusters):
            clusterSize = len(cluster)
            if i in cluster:
                # compute a(obj)
                if clusterSize > 1:
                    a[i] = np.sum(distMatrix[i][cluster])/(clusterSize-1)
                else:
                    a[i] = 0
            else:
                # compute b(obj)
                tempb = np.sum(distMatrix[i][cluster])/clusterSize
                if tempb < b[i]: 
                    b[i] = tempb
                
    for i in range(N):
        silhouette[i] = 0 if a[i] == 0 else (b[i]-a[i])/np.max([a[i], b[i]])
    
    return silhouette

# Computes Silhouette Coefficient for the dataset with respect to the given clustering and distance matrix
def silhouette(dataset, clusters, distMatrix):
    return np.mean(silhouetteCoefficient(dataset, clusters, distMatrix))

if __name__ == '__main__':
    dataset = load_data('dataset')
    KMeans(dataset)