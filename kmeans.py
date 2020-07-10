"""Demonstration of K-Means classifier.

Uses the optdigits dataset. With k = 10 centroids the accuracy reaches 72%.
With k = 30 the accuracy reaches 80%.
"""

import numpy as np
import pandas as pd
import math
import random
import argparse
from sklearn.metrics import confusion_matrix
from PIL import Image
import matplotlib.pyplot as plt


def centroids_predictions(classes, clusters, centroids):
    """Return predictions based on cluster and centroids."""
    cluster_classes = []
    res = {}
    for i in range(len(clusters)):
        frequencies = classes.loc[clusters[i]].value_counts()
        class_type = frequencies.index[0]
        # If class type is already in set, just continue to next cluster.
        if class_type in cluster_classes:
            continue
        cluster_classes.append(class_type)
        res[class_type] = centroids[i]
    return res


def recalculate_centroids(df, clusters):
    """Return new centroids based on clusters."""
    centroids = [[] for x in range(len(clusters))]
    for i in range(len(clusters)):
        centroids[i] = np.array(df.loc[clusters[i]].mean().values)
    return np.array(centroids)


def get_accuracy(actual_class, predicted_class):
    """Return accuracy."""
    correct = 0.0
    for i in range(len(actual_class)):
        if actual_class[i] == predicted_class[i]:
            correct += 1
    return correct / len(actual_class)


def calculate_entropy(classes, cluster):
    """Return entropy of cluster."""
    cluster_clasess = classes.loc[cluster]
    entropy_sum = 0
    mi = len(cluster_clasess)
    for class_type in cluster_clasess.unique():
        mij = len((cluster_clasess == class_type).to_numpy().nonzero()[0])
        entropy_sum += (mij / mi) * math.log2(mij / mi)

    return entropy_sum * -1


def calculate_mean_entropy(classes, clusters):
    """Return mean entropy of cluster."""
    mean_sum = 0
    for i in range(len(clusters)):
        mean_sum += (len(clusters[i]) / len(classes)) * calculate_entropy(
            classes, clusters[i]
        )
    return mean_sum


def generate_pairs(lst):
    """Return all possible pars of a list."""
    pairs = []
    for i in range(len(lst)):
        for j in range(i):
            if i != j:
                pairs.append([i, j])
    return pairs


def calculate_MSS(centroids):
    """Return mss of centroids."""
    pairs = generate_pairs(centroids)
    sum_MSS = 0
    k = len(centroids)
    for pair in pairs:
        sum_MSS += (
            euclidian_distance_points(centroids[pair[0]], centroids[pair[1]])
            ** 2
        )
    return sum_MSS / (k * (k - 1) / 2)


def calculate_MSE_cluster(cluster, centroid):
    """Return MSE from cluster to centroid."""
    return np.sum(np.sum(np.power((cluster - centroid), 2), axis=1)) / len(
        cluster
    )


# calcualte the average mse based on the mse of each cluster and centroids
def calculate_average_MSE(df, clusters, centroids):
    """Return average MSE from cluster to centroids."""
    sum_MSE = 0
    for i in range(len(clusters)):
        sum_MSE += calculate_MSE_cluster(df.iloc[clusters[i]], centroids[i])
    return sum_MSE / len(clusters)


def get_euclidian_distance(cluster, centroid):
    """Return distance between cluster and centroids."""
    return np.power(np.sum(np.power((cluster - centroid), 2), axis=1), 0.5)


def euclidian_distance_points(p1, p2):
    """Return distance between two points."""
    return math.sqrt(np.sum(np.power((p1 - p2), 2), axis=0))


def random_centroids(k, dim, minimum, maximum):
    """Return random centroids."""
    centroids = []
    for i in range(k):
        random_center = np.array(
            [random.randint(minimum, maximum) for x in range(dim)]
        )
        centroids.append(random_center)
    return np.array(centroids)


def main(train_file, test_file, k):
    headers = ["f{}".format(x) for x in range(1, 65)]
    headers.append("class")
    train_df = pd.read_csv(train_file, names=headers)
    test_df = pd.read_csv(test_file, names=headers)
    epochs = 1
    train_centroids = []
    train_clusters = []
    min_MSE = float("inf")
    # Restart epochs many times. Keep best result.
    for epoch in range(epochs):
        print("Epoch", epoch)
        # Get random centroids from the train data.
        new_centroids = train_df.drop("class", axis=1).sample(k).values
        # Get random centroids to compare against.
        old_centroids = random_centroids(k=k, dim=64, minimum=0, maximum=16)
        # Repeat until the mean value of the centroid does not change much.
        while abs(np.mean(new_centroids - old_centroids)) > 0.001:
            clusters = [[] for x in range(len(new_centroids))]
            clusters_final = []
            distances = []
            # Get the distance from all train data to each centroid.
            for i in range(len(new_centroids)):
                distances.append(
                    get_euclidian_distance(
                        train_df.drop("class", axis=1), new_centroids[i]
                    )
                )
            # Get which data point is closer to which centroid.
            closest_centroids = np.argmin(np.array(distances), axis=0)
            # Set each cluster's members to the data points they are closer
            # to.
            for i in range(len(clusters)):
                clusters[i] = np.where(closest_centroids == i)[0]
            # Remove any empty clusters from calculations since they are not
            # going to be updated anymore.
            empty_clusters = []
            for j in range(len(clusters)):
                if clusters[j].any():
                    clusters_final.append(clusters[j])
                else:
                    empty_clusters.append(j)
            new_centroids = np.delete(new_centroids, empty_clusters, axis=0)
            # Save the centroids to calculate the man for next loop.
            old_centroids = new_centroids.copy()
            # Recalculate new centroids based on the clusters.
            new_centroids = recalculate_centroids(
                train_df.drop("class", axis=1), clusters_final
            )
            # Ff the average mse is the minimum, saved that cluster.
            avg_MSE = calculate_average_MSE(
                train_df.drop("class", axis=1), clusters_final, new_centroids
            )
            if avg_MSE < min_MSE:
                min_MSE = avg_MSE
                train_centroids = new_centroids.copy()
                train_clusters = clusters_final.copy()
    # With the lowest average mse clusters, calculate metrics.
    avg_MSE = calculate_average_MSE(
        train_df.drop("class", axis=1), train_clusters, train_centroids
    )
    mss = calculate_MSS(train_centroids)
    mean_entropy = calculate_mean_entropy(train_df["class"], train_clusters)
    print("average MSE", avg_MSE, "MSS", mss, "Mean Entropy", mean_entropy)
    # Fet the predicted class based on each cluster class frequency.
    prediction_centroids = centroids_predictions(
        train_df["class"], train_clusters, train_centroids
    )
    prediction_classes = []
    test_data = test_df.drop("class", axis=1)
    # Classify the test data based on its distance to each centroid.
    for index in test_data.index:
        min_dist = float("inf")
        class_type = -1
        for key in prediction_centroids:
            dist = np.power(
                np.sum(
                    np.power(
                        (test_data.iloc[index] - prediction_centroids[key]), 2
                    )
                ),
                0.5,
            )
            if dist < min_dist:
                min_dist = dist
                class_type = key
        prediction_classes.append(class_type)
    # Print confusion matrix
    print("Confusion matrix")
    cm = confusion_matrix(test_df["class"].values, prediction_classes)
    print(cm)
    print(get_accuracy(test_df["class"].values, prediction_classes))
    # show the graphical representation of each centroid
    plt.gray()
    for key in prediction_centroids:
        t = prediction_centroids[key] * 15
        t.resize((8, 8))
        im = Image.fromarray(t)
        plt.figure()
        plt.title(key)
        plt.imshow(im)

    plt.show()


if __name__ == "__main__":
    random.seed(2814)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-train",
        dest="train",
        help="optidigits train file",
        default="optdigits.train",
    )
    parser.add_argument(
        "-test",
        dest="test",
        help="optidigits test file",
        default="optdigits.test",
    )
    parser.add_argument(
        "-k", dest="k", help="number of centroids", default=10,
    )
    args = parser.parse_args()
    main(args.train, args.test, int(args.k))
