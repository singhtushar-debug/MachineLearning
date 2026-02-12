import random
import math


class K_Means:
    def __init__(self, k=3, iterations=100):
        """
        Initializes the KMeanss model with k clusters and iteration limits.

        Args:
            k (int): Number of clusters to generate. Default to 3
            iterations (int): Max number of iterations to calculate and update the centroids.
        """
        self.k = k
        self.iterations = iterations
        self.centroids = []
        self.clusters = []

    def euclidean_distance(self, p1, p2):
        """
        Calculates the Euclediean distance between two points in n-dimensional space.

        Args:
            p1 (list): First point.
            p2 (list): Second point.
        
        Returns:
            dist (float): The Euclidean distance between p1 and p2.
        """
        dist = 0
        for i in range(len(p1)):
            dist += (p1[i] - p2[i]) ** 2
        return math.sqrt(dist)

    def train_model(self, data):
        """
        Trains the model by finding the optimal centroids for the provided data.

        The process :
        1.Intialization: Randomly seleck k points as starting centroids.
        2.Assginment: Assign each point to the nearest centroid.
        3.Update: Re-calculate the centroid as the mean of all points in that cluster.
        4.Repeat: Repeat until centroids stop moving or max-iterations is reached.

        Args:
            data (list of lists): The input numeric dataset where each sub-list is a point.
        """
        self.centroids = [list(point) for point in random.sample(data, self.k)]

        for i in range(self.iterations):
            self.clusters = [[] for _ in range(self.k)]

            for point in data:
                distances = [self.euclidean_distance(point, c) for c in self.centroids]
                closest_ind = distances.index(min(distances))
                self.clusters[closest_ind].append(point)

            prev_centroids = [list(c) for c in self.centroids]

            for j in range(self.k):
                if not self.clusters[j]:
                    continue

                num_dimensions = len(data[0])
                new_centroid = []
                for d in range(num_dimensions):
                    dimension_sum = sum(p[d] for p in self.clusters[j])
                    new_centroid.append(dimension_sum / len(self.clusters[j]))

                self.centroids[j] = new_centroid

            if prev_centroids == self.centroids:
                print(f"Converged at iteration {i}")
                break

    def predict(self, point):
        """
        Predicts which cluster a new data point belongs to.

        Args:
            point (list): A single data point (list of coordinates).

        Returns:
            int: The index of the nearest cluster centroid.
        """
        distances = [self.euclidean_distance(point, c) for c in self.centroids]
        return distances.index(min(distances))
