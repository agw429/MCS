from typing import List

class Solution:
    def calc_euclidean_dist(self, x1, x2):
        return sum((x1[i] - x2[i]) ** 2 for i in range(len(x1))) ** 0.5

    def calc_mean(self, arr):
        return sum(arr) / len(arr)

    def check_inputs(self, X: List[List[float]], K: int):
        if (K > len(X)):
            print("Cannot form more clusters than initial points. Setting K to {}.".format(len(X)))
            K = len(X)
        elif (K < 1):
            print("Cannot form less than 1 cluster. Setting K to 1.")
            K = 1
        return (X, K)

    def hclus_single_link(self, X: List[List[float]], K: int) -> List[int]:
        """Single link hierarchical clustering
        Args:
          - X: 2D input data
          - K: the number of output clusters
        Returns:
          A list of integers (range from 0 to K - 1) that represent class labels.
          The number does not matter as long as the clusters are correct.
          For example: [0, 0, 1] is treated the same as [1, 1, 0]"""

        X, K = self.check_inputs(X, K)

        # To start, each data point is a cluster
        clusters = [[i] for i in range(len(X))]

        # Until we have the correct number of clusters, iteratively combine them
        while len(clusters) > K:

            # First we find the two closest clusters
            min_dist = float('inf')
            closest_clusts = (0, 0)
            for i in range(len(clusters)):
                for j in range(i+1, len(clusters)):

                    # We must calculate the dist between all pairs of clusters using single link
                    dist = min([self.calc_euclidean_dist(X[p], X[q])
                                   for p in clusters[i] for q in clusters[j]])
                    if dist < min_dist:
                        min_dist = dist
                        closest_clusts = (i, j)

            # At each iteration, we merge the two closest clusters
            clusters[closest_clusts[0]] += clusters[closest_clusts[1]]
            del clusters[closest_clusts[1]]

        # Finally, we assign cluster labels to the data points
        labels = [0] * len(X)
        for i, c in enumerate(clusters):
            for p in c:
                labels[p] = i
        return labels


    def hclus_average_link(self, X: List[List[float]], K: int) -> List[int]:
        """Average link hierarchical clustering
        Args:
          - X: 2D input data
          - K: the number of output clusters
        Returns:
          A list of integers (range from 0 to K - 1) that represent class labels.
          The number does not matter as long as the clusters are correct.
          For example: [0, 0, 1] is treated the same as [1, 1, 0]"""
        
        X, K = self.check_inputs(X, K)
        
        # To start, each data point is a cluster
        clusters = [[i] for i in range(len(X))]

        # Until we have the correct number of clusters, iteratively combine them
        while len(clusters) > K:

            # First we find the two closest clusters
            min_dist = float('inf')
            closest_clusts = (0, 0)
            for i in range(len(clusters)):
                for j in range(i+1, len(clusters)):

                    # We must calculate the dist between all pairs of clusters using average link
                    dist = self.calc_mean([self.calc_euclidean_dist(X[p], X[q]) for p in clusters[i] for q in clusters[j]])
                    if dist < min_dist:
                        min_dist = dist
                        closest_clusts = (i, j)
                        
            # At each iteration, we merge the two closest clusters
            clusters[closest_clusts[0]] += clusters[closest_clusts[1]]
            del clusters[closest_clusts[1]]

        # Finally, we assign cluster labels to the data points
        labels = [0] * len(X)
        for i, c in enumerate(clusters):
            for p in c:
                labels[p] = i
        return labels


    def hclus_complete_link(self, X: List[List[float]], K: int) -> List[int]:
        """Complete link hierarchical clustering
        Args:
          - X: 2D input data
          - K: the number of output clusters
        Returns:
          A list of integers (range from 0 to K - 1) that represent class labels.
          The number does not matter as long as the clusters are correct.
          For example: [0, 0, 1] is treated the same as [1, 1, 0]"""

        X, K = self.check_inputs(X, K)

        # To start, each data point is a cluster
        clusters = [[i] for i in range(len(X))]

        # Until we have the correct number of clusters, iteratively combine them
        while len(clusters) > K:

            # First we find the two closest clusters
            min_dist = float('inf')
            closest_clusts = (0, 0)
            for i in range(len(clusters)):
                for j in range(i+1, len(clusters)):

                    # We must calculate the dist between all pairs of clusters using complete link
                    dist = max([self.calc_euclidean_dist(X[p], X[q])
                                for p in clusters[i] for q in clusters[j]])
                    if dist < min_dist:
                        min_dist = dist
                        closest_clusts = (i, j)

            # At each iteration, we merge the two closest clusters
            clusters[closest_clusts[0]] += clusters[closest_clusts[1]]
            del clusters[closest_clusts[1]]

        # Finally, we assign cluster labels to the data points
        labels = [0] * len(X)
        for i, c in enumerate(clusters):
            for p in c:
                labels[p] = i
        return labels
