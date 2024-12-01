import numpy as np
import torch
import tkinter as tk
from random import randrange as random, choice
from time import time
from warnings import filterwarnings
filterwarnings("ignore", category=DeprecationWarning)

START_VERTICES = 20
XY_MIN = -5000
XY_MAX = 5000
CONSTANT = 14
OFFSET = 50
MAX_CLUSTERS = 20
DISTANCE_THRESHOLD = 150
DISPLAY_DIAMETER = 3
XY_OFFSET = 100

def create_initials():
    vertices = set()
    while len(vertices) < START_VERTICES:
        x_coord = random(XY_MIN, XY_MAX + 1)
        y_coord = random(XY_MIN, XY_MAX + 1)
        vertices.add((x_coord, y_coord))
    return list(vertices)

def generate_vertex(current):
    reference = choice(current)
    x_min = max(-XY_OFFSET, XY_MIN - reference[0])
    x_max = min(XY_OFFSET+1, XY_MAX - reference[0] + 1)
    y_min = max(-XY_OFFSET, XY_MIN - reference[1])
    y_max = min(XY_OFFSET+1, XY_MAX - reference[1] + 1)
    x_offset = random(x_min, x_max)
    y_offset = random(y_min, y_max)
    return (reference[0] + x_offset, reference[1] + y_offset)

def display_vertices(canvas, vertices, clusters, centroids):
    import random as rnd
    cluster_colors = ["#%06x" % rnd.randint(0, 0xFFFFFF) for _ in range(len(centroids))]

    for i, (x, y) in enumerate(vertices):
        canvas_x = ((x + XY_MAX) / CONSTANT) + OFFSET
        canvas_y = ((y + XY_MAX) / CONSTANT) + OFFSET
        color = cluster_colors[clusters[i]]
        canvas.create_oval(canvas_x - 1, canvas_y - 1, canvas_x + 1, canvas_y + 1, fill=color, outline='')

    for x, y in centroids:
        canvas_x = ((x + XY_MAX) / CONSTANT) + OFFSET
        canvas_y = ((y + XY_MAX) / CONSTANT) + OFFSET
        canvas.create_oval(canvas_x - DISPLAY_DIAMETER, canvas_y - DISPLAY_DIAMETER,
                           canvas_x + DISPLAY_DIAMETER, canvas_y + DISPLAY_DIAMETER, fill='')

    canvas.create_rectangle(OFFSET, OFFSET, (OFFSET + (10000 / CONSTANT)), (OFFSET + (10000 / CONSTANT)), fill='')

# Perform K-means clustering to split clusters
def kmeans_clustering(vertices, n_clusters=2, max_iter=10, tol=1e-4):
    data = torch.tensor(vertices, dtype=torch.float32)
    centroids = data[torch.randperm(len(data))[:n_clusters]]

    for _ in range(max_iter):
        distances = torch.cdist(data, centroids)
        labels = torch.argmin(distances, dim=1)
        new_centroids = torch.stack([data[labels == j].mean(dim=0) for j in range(n_clusters)])

        if torch.allclose(centroids, new_centroids, atol=tol):
            break
        centroids = new_centroids

    return centroids.cpu().numpy(), labels.cpu().numpy()

# Divisive clustering implementation
def divisive_clustering(points):
    points = np.array(points)  #points is a NumPy array
    clusters = [points]
    centroids = [np.mean(points, axis=0)]
    final_max_average_distance = 0
    
    while len(clusters) < MAX_CLUSTERS:
        max_variance = 0
        cluster_to_split = None
        index_to_split = -1

        for i, cluster_points in enumerate(clusters):
            if cluster_points.shape[0] > 1:
                variance = np.mean(np.linalg.norm(cluster_points - centroids[i], axis=1))
                if variance > max_variance:
                    max_variance = variance
                    cluster_to_split = cluster_points
                    index_to_split = i

        if cluster_to_split is None:
            break

        sub_centroids, sub_labels = kmeans_clustering(cluster_to_split.tolist())
        del clusters[index_to_split]
        del centroids[index_to_split]

        cluster1 = cluster_to_split[sub_labels == 0]
        cluster2 = cluster_to_split[sub_labels == 1]

        clusters.append(cluster1)
        clusters.append(cluster2)
        centroids.append(np.mean(cluster1, axis=0))
        centroids.append(np.mean(cluster2, axis=0))

        all_within_threshold = True
        max_average_distance = 0
        for cluster_points, centroid in zip(clusters, centroids):
            if len(cluster_points) > 0:
                avg_distance = np.mean(np.linalg.norm(cluster_points - centroid, axis=1))
                max_average_distance = max(max_average_distance, avg_distance)
                if avg_distance > DISTANCE_THRESHOLD:
                    all_within_threshold = False
        
        final_max_average_distance = max_average_distance

        if all_within_threshold:
            break

    labels = np.zeros(len(points), dtype=int)
    for idx, cluster_points in enumerate(clusters):
        indices = np.where(np.isin(points.view([('', points.dtype)] * points.shape[1]),
                                    cluster_points.view([('', cluster_points.dtype)] * cluster_points.shape[1])))[0]
        labels[indices] = idx

    centroids = [centroid.tolist() for centroid in centroids]
    return labels, centroids, final_max_average_distance

if __name__ == "__main__":
    start = time()
    vertices = create_initials()
    for _ in range(40000):
        vertices.append(generate_vertex(vertices))

    labels, centroids, final_max_average_distance = divisive_clustering(vertices)
    end = time()
    print(f"max average distance from centroid: {final_max_average_distance:.2f}, number of centorid: {len(centroids)}")
    print(f"Run time: {end - start:.2f} seconds")
    
    root = tk.Tk()
    canvas = tk.Canvas(root, width=(2 * OFFSET + 10000 / CONSTANT), height=(2 * OFFSET + 10000 / CONSTANT), bg='white')
    canvas.pack()
    display_vertices(canvas, vertices, labels, centroids)
    root.mainloop()

