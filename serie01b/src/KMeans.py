import pandas as pd
from pandas import DataFrame
from sklearn.base import ClusterMixin, BaseEstimator
from sklearn.neighbors import DistanceMetric
import numpy as np

class KMeans(ClusterMixin, BaseEstimator):
    k_clusters: int
    dist: DistanceMetric

    max_iterations: int
    random_seed: int

    X: DataFrame
    centroids: np.ndarray
    labels: np.ndarray

    # Possible termination criteria:
    # * √ Fixed number of iterations
    # * √ Small or no change of the cluster centers
    # * Small or no change of the cluster assignment
    # * Small clustering error EK (see next slide) or small decrease of EK

    # Possible choices of initial cluster centers:
    # √ Random choice of K elements of S
    # * Random generation of K elements x∈R^n in the same cuboid as S
    # * As above but with a minimum distance between the K elements

    # Possible post-processing steps:
    # * Merge small clusters
    # * Split large clusters, for example if they have a large variance
    def __init__(self, k_clusters: int, distance_method: str = 'euclidean', max_iterations=200, random_seed=4242):
        self.k_clusters = k_clusters
        self.dist = DistanceMetric.get_metric(distance_method)
        self.max_iterations = max_iterations
        self.random_seed = random_seed
        pass

    def _initialize_centroids(self):
        return self.X.sample(self.k_clusters, random_state=self.random_seed).to_numpy()

    def _assign_to_clusters(self, X: DataFrame, centroids: np.ndarray) -> np.ndarray:
        # 1. Compute the distance between each row and each centroids
        #       Produces a matrix, with
        #                           each row being the instances of X
        #                           each column is the distance we the centroid i
        distances = self.dist.pairwise(X, centroids)
        # 2. pick the closest
        #       In each row, select the smallest distance
        return np.argmin(distances, axis=1)

    def _compute_new_centroids(self, labels) -> np.ndarray:
        # in each cluster/labels, look for the best centroid
        centroids = np.zeros((self.k_clusters, self.X.shape[1]))
        for label_i in range(self.k_clusters):
            # TODO check if empty dataframe, might be in trouble here
            centroids[label_i, :] = self.X[labels == label_i].mean()

        return centroids

    # the label, Y is ignored for KMeans
    def fit(self, X):
        self.X = X
        self.centroids = self._initialize_centroids()
        for _ in range(self.max_iterations):
            initial_centroids = self.centroids
            self.labels = self._assign_to_clusters(X, initial_centroids)
            self.centroids = self._compute_new_centroids(self.labels)
            if np.all(initial_centroids == self.centroids):
                return self

        # reached the max number of iterations, therefore, recompute the labels
        self.labels = self._assign_to_clusters(X, self.centroids)
        return self

    def dunn_index_score(self) -> float:
        def _shortest_dist_between_cluster(X: DataFrame, dist: DistanceMetric, labels: np.ndarray, label_i: int, label_j: int) -> float:
            return np.min(dist.pairwise(X[labels == label_i], X[labels == label_j]))

        # TODO parallelize
        smallest_cluster_dist = np.min([_shortest_dist_between_cluster(self.X, self.dist, self.labels, label_i, label_j)
            for label_i in range(self.k_clusters) for label_j in range(label_i, self.k_clusters) if label_i != label_j])

        def _cluster_diameter(X: DataFrame, dist: DistanceMetric, labels: np.ndarray, label: int) -> float:
            return np.max(dist.pairwise(X[labels == label]))

        # TODO parallelize
        # aka delta_max
        max_cluster_diameter = np.max([_cluster_diameter(self.X, self.dist, self.labels, label_i)
                                       for label_i in range(self.k_clusters)])

        return smallest_cluster_dist / max_cluster_diameter

    def c_index_score(self) -> float:
        # https://cran.r-project.org/web/packages/clusterCrit/vignettes/clusterCrit.pdf
        # Page 9
        # C-Index = (S_W - S_min) / (S_max - S_min)
        #         = (Gamma - min) / (max - min)
        # with:
        #   N_W: the total number of such pairs
        #       N_W = (sum k=1 (n_k * (n_k - 1) / 2) to K)
        #           = (sum k=1 (n_k^2 - N) to K) / 2
        #   N_T: the total number of pairs of distinct points
        #       N_T = N * (N - 1) / 2 = N_W + (sum k < k' (n_k * n_k'))
        #       N_T = N_W + N_B
        #   N_B: the number of pairs constituted of points which do not belong to the same cluster
        #       N_B = sum k < k' (n_k * n_k')
        #   S_W: sum of the N_W distances between all the pairs of points inside each cluster.
        #        Also called gamma in the slides
        #   S_min: sum of the N_W smallest distances between all the pairs of points in the entire data set.
        #          There are N_T such pairs: one takes the sum of the N_W smallest values
        #   S_max: sum of the N_W largest distances between all the pairs of points in the entire data set.
        #          There are N_T such pairs: one takes the sum of the N_W largest values

        # 1. Gamma/S_W
        #   1.1 For each cluster, sum the distance of each instances inside that cluster
        #   1.2 Sum all those sum

        def _sum_distance_in_cluster(X: DataFrame, dist: DistanceMetric, labels: np.ndarray, label: int) -> float:
            return np.sum(dist.pairwise(X[labels == label]))


        # TODO parallelize or there is most likely a way to compute all the distance once,
        #       and then pick the distances with the help of the labels (not the time to figure it out unfortunately)
        # divide by 2, as the matrix is symmetric
        #   (distance is computed between `a` to `b` and `b` to `a` in a different position)
        gamma = np.sum([_sum_distance_in_cluster(self.X, self.dist, self.labels, label_i) for label_i in range(self.k_clusters)]) / 2

        # Only select the lower part of the matrix
        low_matrix_selection = np.tril_indices(self.X.shape[0])
        # The selection will also flatten the matrix
        distance_X = self.dist.pairwise(self.X)[low_matrix_selection]

        per_label_counts = pd.DataFrame(self.labels).value_counts()
        alpha = int((per_label_counts * (per_label_counts - 1) / 2).sum())

        s_min = distance_X[np.argpartition(distance_X, alpha)[:alpha]].sum()
        s_max = distance_X[np.argpartition(-distance_X, alpha)[:alpha]].sum()

        return (gamma - s_min) / (s_max - s_min)

    def predict(self, x: DataFrame):
        return self._assign_to_clusters(x, self.centroids)
