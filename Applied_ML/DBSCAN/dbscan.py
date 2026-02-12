import math
class DBSCAN:
    def __init__(self,eps = 1,mnPts = 3):
        """
        Initializes the DBSCAN clusterer.
        
        Args:
            eps (float): The distance threshold (epsilon).
            mnPts (int): Minimum number of points required to form a dense region.
        """
        self.eps = eps
        self.mnPts = mnPts
        self.labels = []

    def euclidean_distance(self,p1,p2):
        """
        Calculates the euclidean distance between two given points.

        Args:
            p1 (list):First point.
            p2 (list):Second point.
        
        Returns:
            float: The euclidean distance between p1 and p2.
        """
        dist = 0
        for i in range(len(p1)):
            dist += (p1[i] - p2[i])**2
        return math.sqrt(dist)
    
    def get_neighbors(self,data,point_ind):
        """
        Finds the indices of all points within the eps neighborhood.

        Args:
            data (list of lists): The input dataset.
            point_ind (int): The index of the point to search around.
        
        Returns:
            neighbors (list[int]): A list of indices representing neighboring points.
        """
        neighbors = []
        for i,point in enumerate(data):
            if self.euclidean_distance(point,data[point_ind]) < self.eps:
                neighbors.append(i)
        
        return neighbors
    
    def train_model(self,data):
        """
        Performs DBSCAN clustering on the provided dataset.

        Iterates through each point, identifies core points and expands clusters until all reachable points are assigned.

        Args:
            data (list of lists): The dataset to cluster.

        Returns:
            labels (list[int]): The calculated labels(cluster_id) for each point.
        """
        self.labels = [None]*len(data)
        cluster_id = 0

        for i in range(len(data)):
            if self.labels[i] is not None:
                continue

            neighbors = self.get_neighbors(data,i)

            if len(neighbors) < self.mnPts:
                self.labels[i] = -1
            else:
                self.labels[i] = cluster_id
                self.expand_cluster(data,neighbors,cluster_id)
                cluster_id += 1
            
        return self.labels
    
    def expand_cluster(self,data,neighbors,cluster_id):
        """
        Expands a cluster from a seed core point using a breadth-first approach.

        This method updates 'Noise' points to 'Border' points and discovers new core points to add the search queue.

        Args:
            data (list of lists): The input dataset.
            neighbors (list[ind]): Initial neighbors of the core point.
            cluster_id (int): The current cluster label being assigned.
        """
        i = 0
        while i < len(neighbors):
            neighbor_ind = neighbors[i]

            if self.labels[neighbor_ind] == -1:
                self.labels[neighbor_ind] = cluster_id
            
            elif self.labels[neighbor_ind] is None:
                self.labels[neighbor_ind] = cluster_id

                new_neighbors = self.get_neighbors(data,neighbor_ind)

                if len(new_neighbors) >= self.mnPts:
                    neighbors.extend(new_neighbors)
            
            i += 1
