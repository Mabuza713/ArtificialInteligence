import random

def euclidean_distance(x1, x2):
    return ((x1[0] - x2[0]) ** 2 + (x1[1] - x2[1]) ** 2) ** 0.5

class Kmeans:
    def __init__(self, k=5, max_iter=300):
        self.k = k
        self.max_iter = max_iter
        self.centroids = None
        self.labels = None
        self.inertia_ = None

    def fit(self, X):
        #standarization
        n = len(X)

        x0_vals = [p[0] for p in X]
        x1_vals = [p[1] for p in X]

        mean_0 = sum(x0_vals) / n
        mean_1 = sum(x1_vals) / n

        std_0 = (sum([(x - mean_0) ** 2 for x in x0_vals]) / n) ** 0.5
        std_1 = (sum([(x - mean_1) ** 2 for x in x1_vals]) / n) ** 0.5

        X_scaled = []
        for p in X:
            scaled_point = ((p[0] - mean_0) / std_0,(p[1] - mean_1) / std_1)
            X_scaled.append(scaled_point)

        self.centroids = random.sample(X_scaled, self.k)

        for _ in range(self.max_iter):
            labels = []
            for i in range(self.k):
                labels.append([])

            for point in X_scaled:
                distances = [euclidean_distance(point, centroid) for centroid in self.centroids]
                labels[distances.index(min(distances))].append(point)

            new_centroids = []
            for i in range(self.k):
                if len(labels[i]) == 0:
                    new_centroids.append(self.centroids[i])
                    continue

                new_centroid = (sum([x[0] for x in labels[i]]) / len(labels[i])
                                    , sum([x[1] for x in labels[i]]) / len(labels[i]))

                new_centroids.append(new_centroid)

            if new_centroids == self.centroids:
                break

            self.centroids = new_centroids

        # calc loss
        for i, centroid in enumerate(self.centroids):
            loss = 0
            for point in labels[i]:
                loss += euclidean_distance(point, centroid)


        return self.centroids, labels, loss