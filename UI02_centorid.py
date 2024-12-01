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
NUM_CLUSTERS = 13
MAX_CLUSTERS = 25
DYSPLAY_DIAMETER = 3
MIN_DISTANCE_BETWEEN_CENTROIDS = 150
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
        radius = 1
        color = cluster_colors[clusters[i]]
        canvas.create_oval(canvas_x - radius, canvas_y - radius,
                           canvas_x + radius, canvas_y + radius,
                           fill=color, outline='')

    # Plot centroids in green
    for x, y in centroids:
        canvas_x = ((x + XY_MAX) / CONSTANT) + OFFSET
        canvas_y = ((y + XY_MAX) / CONSTANT) + OFFSET
        canvas.create_oval(canvas_x - DYSPLAY_DIAMETER, canvas_y - DYSPLAY_DIAMETER, canvas_x + DYSPLAY_DIAMETER, canvas_y + DYSPLAY_DIAMETER, fill='') #, outline='#DFDFDF'

    # Draw boundary box
    canvas.create_rectangle(OFFSET, OFFSET, (OFFSET + (10000 / CONSTANT)), (OFFSET + (10000 / CONSTANT)), fill='')


def kmeans_clustering(vertices, n_clusters=20, max_iter=300, tol=1e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = torch.tensor(vertices, dtype=torch.float32).to(device)
    n_samples, n_features = data.shape
    centroids = data[torch.randperm(n_samples)[:n_clusters]]

    for _ in range(max_iter):
        distances = torch.cdist(data, centroids)
        labels = torch.argmin(distances, dim=1)
        new_centroids = torch.stack([data[labels == j].mean(dim=0) if (labels == j).any() else centroids[j] for j in range(n_clusters)])

        if torch.allclose(centroids, new_centroids, atol=1e-8):
            break
        centroids = new_centroids

    return centroids.cpu().numpy(), labels.cpu().numpy()


def evaluate_clusters(vertices, labels, centroids, threshold=500):
    global max_average_distance
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = torch.tensor(vertices, dtype=torch.float32).to(device)
    labels = torch.tensor(labels, dtype=torch.long).to(device)
    centroids = torch.tensor(centroids, dtype=torch.float32).to(device)
    n_clusters = centroids.shape[0]

    while True:
        max_average_distance = 0
        for i in range(n_clusters):
            cluster_points = data[labels == i]
            if len(cluster_points) == 0:
                continue
            distances = torch.norm(cluster_points - centroids[i], dim=1)
            average_distance = torch.mean(distances).item()
            max_average_distance = max(max_average_distance, average_distance)
            evaluate_clusters.max_average_distance = max_average_distance

        if max_average_distance > threshold:
            n_clusters += 1
            centroids, labels = kmeans_clustering(vertices, n_clusters=n_clusters)
        else:
            break

    return centroids, labels


def remove_close_centroids(centroids, min_distance=MIN_DISTANCE_BETWEEN_CENTROIDS):
    centroids = torch.tensor(centroids, dtype=torch.float32)
    distance_matrix = torch.cdist(centroids, centroids)

    # Set diagonal to a high value to ignore self-distance
    distance_matrix += torch.eye(len(centroids)) * (min_distance + 1)

    to_remove = set()
    for i in range(len(centroids)):
        close_centroids = torch.where(distance_matrix[i] < min_distance)[0]
        if len(close_centroids) >= 2:
            to_remove.add(i)

    # Remove centroids that are too close
    filtered_centroids = [centroids[i].tolist() for i in range(len(centroids)) if i not in to_remove]
    return filtered_centroids


def cluster_vertices(vertices):
    while True:
        # initial NUM_CLUSTERS
        centroids, labels = kmeans_clustering(vertices, n_clusters=NUM_CLUSTERS)

        centroids, labels = evaluate_clusters(vertices, labels, centroids)

        centroids = remove_close_centroids(centroids)

        if len(centroids) <= MAX_CLUSTERS:
            break
        else:
            print(f"Number of clusters ({len(centroids)}) exceeded the maximum allowed")\

    return centroids, labels


if __name__ == "__main__":
    start = time()
    # Initialize vertices
    vertices = create_initials()
    for _ in range(40000):
        vertices.append(generate_vertex(vertices))

    # Cluster vertices
    centroids, labels = cluster_vertices(vertices)
    print(f"Max average distance from medoid: {evaluate_clusters.max_average_distance:.2f}, number of clusters: {len(centroids)}")
    
    # Display solution
    root = tk.Tk()  
    canvas = tk.Canvas(root, width=(2 * OFFSET + 10000 / CONSTANT), height=(2 * OFFSET + 10000 / CONSTANT), bg='white')
    canvas.pack()
    display_vertices(canvas, vertices, labels, centroids)

    end = time()
    print(f"Run time: {end - start:.2f} seconds")
    print(f"Final max average distance from centroid: {evaluate_clusters.max_average_distance:.2f}")
    root.mainloop()
