import numpy as np
import torch
from random import randrange as random, choice
from time import time
import tkinter as tk
from warnings import filterwarnings
filterwarnings("ignore", category=DeprecationWarning)


# Constants
START_VERTICES = 20
XY_MIN = -5000
XY_MAX = 5000
XY_OFFSET = 100
CONSTANT = 14       
OFFSET = 50
NUM_CLUSTERS = 13
MAX_CLUSTERS = 25  # Maximum allowed number of clusters
# MIN_DISTANCE_BETWEEN_MEDOIDS = 150
DYSPLAY_DIAMETER = 3
# global max_average_distance
 

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


def display_vertices(canvas, vertices, clusters, medoids):
    import random as rnd
    cluster_colors = ["#%06x" % rnd.randint(0, 0xFFFFFF) for _ in range(len(medoids))]

    for i, (x, y) in enumerate(vertices):
        canvas_x = ((x + XY_MAX) / CONSTANT) + OFFSET
        canvas_y = ((y + XY_MAX) / CONSTANT) + OFFSET
        radius = 1
        color = cluster_colors[clusters[i]]
        canvas.create_oval(canvas_x - radius, canvas_y - radius,
                           canvas_x + radius, canvas_y + radius,
                           fill=color, outline='')

    for x, y in medoids:
        canvas_x = ((x + XY_MAX) / CONSTANT) + OFFSET
        canvas_y = ((y + XY_MAX) / CONSTANT) + OFFSET
        canvas.create_oval(canvas_x - DYSPLAY_DIAMETER, canvas_y - DYSPLAY_DIAMETER, canvas_x + DYSPLAY_DIAMETER, canvas_y + DYSPLAY_DIAMETER, fill='') #, outline='#DFDFDF'

    canvas.create_rectangle(OFFSET, OFFSET, (OFFSET + (10000 / CONSTANT)), (OFFSET + (10000 / CONSTANT)), fill='')


def kmedoids_clustering(vertices, n_clusters=20, max_iter=300):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") #use gpu for calculations, if unable use cpu
    data = torch.tensor(vertices, dtype=torch.float32).to(device)
    n_samples = data.shape[0]
    medoids_indices = torch.randperm(n_samples)[:n_clusters]
    medoids = data[medoids_indices]

    for _ in range(max_iter):
        distances = torch.cdist(data, medoids)
        labels = torch.argmin(distances, dim=1)

        new_medoids = []
        for i in range(n_clusters):
            cluster_points = data[labels == i]
            if len(cluster_points) == 0:
                new_medoids.append(medoids[i])
                continue
            pairwise_distances = torch.cdist(cluster_points, cluster_points)
            medoid_index = torch.argmin(pairwise_distances.sum(dim=1))
            new_medoids.append(cluster_points[medoid_index])

        new_medoids = torch.stack(new_medoids)
        if torch.equal(medoids, new_medoids):
            break
        medoids = new_medoids

    return medoids.cpu().numpy(), labels.cpu().numpy()


def evaluate_clusters(vertices, labels, medoids, threshold=500):
    global max_average_distance
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = torch.tensor(vertices, dtype=torch.float32).to(device)
    labels = torch.tensor(labels, dtype=torch.long).to(device)
    medoids = torch.tensor(medoids, dtype=torch.float32).to(device)
    n_clusters = medoids.shape[0]

    while True: #find max average distance
        max_average_distance = 0
        for i in range(n_clusters):
            cluster_points = data[labels == i]
            if len(cluster_points) == 0:
                continue
            distances = torch.norm(cluster_points - medoids[i], dim=1)
            average_distance = torch.mean(distances).item()
            max_average_distance = max(max_average_distance, average_distance)
            evaluate_clusters.max_average_distance = max_average_distance

        if max_average_distance > threshold:
            n_clusters += 1
            medoids, labels = kmedoids_clustering(vertices, n_clusters=n_clusters)
        else:
            break

    return medoids, labels


# def remove_close_medoids(medoids, min_distance=MIN_DISTANCE_BETWEEN_MEDOIDS):
#     medoids = torch.tensor(medoids, dtype=torch.float32)
#     distance_matrix = torch.cdist(medoids, medoids)

#     # Set diagonal to a high value to ignore self-distance
#     distance_matrix += torch.eye(len(medoids)) * (min_distance + 1)

#     to_remove = set()
#     for i in range(len(medoids)):
#         close_medoids = torch.where(distance_matrix[i] < min_distance)[0]
#         if len(close_medoids) >= 2:
#             to_remove.add(i)

#     # Remove medoids that are too close
#     filtered_medoids = [medoids[i].tolist() for i in range(len(medoids)) if i not in to_remove]
#     return filtered_medoids


def cluster_vertices(vertices):
    while True:
        # cluster
        medoids, labels = kmedoids_clustering(vertices, n_clusters=NUM_CLUSTERS)
        # evaluate
        medoids, labels = evaluate_clusters(vertices, labels, medoids)

        # Remove medoids that are too close
        # medoids = remove_close_medoids(medoids)

        if len(medoids) <= MAX_CLUSTERS:
            break
        else:
            print(f"Number of clusters ({len(medoids)}) exceeded the maximum allowed")

    return medoids, labels


if __name__ == "__main__":
    start = time()
    vertices = create_initials()
    for _ in range(40000):
        vertices.append(generate_vertex(vertices))

    medoids, labels = cluster_vertices(vertices)
    print(f"Max average distance from medoid: {evaluate_clusters.max_average_distance:.2f}, number of clusters: {len(medoids)}")
    root = tk.Tk()   # Display  
    canvas = tk.Canvas(root, width=(2 * OFFSET + 10000 / CONSTANT), height=(2 * OFFSET + 10000 / CONSTANT)) #, bg='white'
    canvas.pack()
    display_vertices(canvas, vertices, labels, medoids)

    root.mainloop()
    end = time()
    print(f"Run time: {end - start:.2f} seconds")
