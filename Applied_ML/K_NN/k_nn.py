import math
class KNN:
    def __init__(self,k = 3):
        """
        Initializes the KNN with a specific k value.

        Args:
            k (int): Number of neighbors. Default to k.
        """
        self.k = k
        self.x_train = None
        self.y_train = None
    
    def train_model(self,x,y):
        """
        Sotres the training data.Since knn is a lazy learning algo and does not perform anything during the training phase.
        It only stores the training data and perform all the calculations during prediction.

        Args:
            x (list of lists): Training features.
            y (list): Training labels.

        """
        self.x_train = x
        self.y_train = y

    def euclidean_distance(self,p1,p2):
        """
        Calculates the euclidean distance between point two points.

        Args:
            p1 (list): First point.
            p2 (list): Second point.
        
        Returns:
            distance (float): Euclidean distance between points p1 and p2.
        """
        dist = 0
        for i in range(len(p1)):
            dist += (p1[i] - p2[i]) ** 2
        return math.sqrt(dist)
    
    def predict(self,x):
        """
        Predicts labels for a provided list of data points.

        Args:
            x (list of lists):New data points to classify.
        
        Returns:
            list: Predicted class labels for each input in x.
        """
        predictions = [self.predict_single(p) for p in x] 
        return predictions
    
    def predict_single(self,p):
        """
        Helper function to predict the label of a single data point.

        The process:
        1.Calculate the distance of every training point.
        2.Sort and identify the 'k' nearest points.
        3.Perform a majority vote on the labels of those points.

        Args:
            x (list): A single data point (feature set).
        
        Returns:
            The most common label among the k-nearest neighbors.
        """
        distances = []

        for i in range(len(self.x_train)):
            dist = self.euclidean_distance(p,self.x_train[i])
            distances.append((dist,self.y_train[i]))

        distances.sort(key=lambda x: x[0])
        k_nearest_label = [label for (_,label) in distances[:self.k]]
        
        counts = {}
        for label in k_nearest_label:
            if label in counts:
                counts[label] += 1
            else:
                counts[label] = 1
        mx_count = -1
        ans_label = None

        for label,count in counts.items():
            if count > mx_count:
                mx_count = count
                ans_label = label
            
        return ans_label