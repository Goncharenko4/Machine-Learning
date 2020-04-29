import numpy as np
from scipy.spatial import distance_matrix
import matplotlib.pyplot as plt
import random
import copy

# Алгоритм K-Means:
def algorithm(X, K):
    nrow = X.shape[0]
    ncol = X.shape[1]

    initial_centroids = np.random.choice(nrow, K, replace=False) # выбераем K случайных точек (начальных центроидов) 
    centroids = X[initial_centroids]
    centroids_old = np.zeros((K, ncol))
    cluster_assignments = np.zeros(nrow)

    while (centroids_old != centroids).any():
        centroids_old = centroids.copy()
        dist_matrix = distance_matrix(X, centroids, p=2) # вычисление расстояний между точками данных и центроидами

        # шаг 1: Найти ближайший центроид для каждой точки данных 
        for i in np.arange(nrow):
            d = dist_matrix[i]
            closest_centroid = (np.where(d == np.min(d)))[0][0]
            cluster_assignments[i] = closest_centroid # связываем точку с ближайшим центроидом

        # шаг 2: пересчитать центроиды
        for k in np.arange(K):
            Xk = X[cluster_assignments == k]
            centroids[k] = np.apply_along_axis(np.mean, axis=0, arr=Xk)

    return (centroids, cluster_assignments)


# Построение графиков:
def take_plot(centroids, cluster_assignments, data):
    for j, core in enumerate(centroids):
        x = [ ]
        y = [ ]
        for i, line in enumerate(data):
            if (cluster_assignments[i] == j):
                # print(line)
                x.append(line[0])
                y.append(line[1])
        plt.scatter(x, y)
    for core in centroids:
        plt.scatter(core[0], core[1], c='black')
    plt.show()

    
# Матрица попарных расстояний:
def distance_table(centroids, cluster_assignments, data):
    i = np.argsort(cluster_assignments)
    lines2 = data[i, :]

    D = (lines2[:, 0][:, np.newaxis] - lines2[:, 0]) ** 2
    D += (lines2[:, 1][:, np.newaxis] - lines2[:, 1]) ** 2
    D = np.sqrt(D)
    plt.figure(figsize=(5, 4))
    plt.imshow(D, cmap='Purples_r', aspect='auto')
    plt.xticks([])
    plt.yticks([])
    plt.colorbar(orientation='vertical', pad=0.06);
    plt.show()    
    
    
# Стратегия выбора числа кластеров
def cluster(data, K):
    minimal_claster_dist = (int(K * (K - 1) / 9)) + K
    distance = (data[:, 0][:, np.newaxis] - data[:, 0]) ** 2
    distance += (data[:, 1][:, np.newaxis] - data[:, 1]) ** 2
    distance = np.sqrt(distance)
    minimal_claster_dist = np.sort(distance.ravel())[minimal_claster_dist]
    minimal_claster_dist *= 0.72
    distance[distance <= minimal_claster_dist] = 0
    dist_copy = np.zeros(K * K).reshape(K, K)
    while True:
        for i in range(K):
            zero_dist = distance[distance [i, :] == 0]
            zero_dist = np.min(zero_dist, axis=0)
            dist_copy[i, :] = zero_dist
        if np.array_equal(dist_copy, distance):
            break
        distance = dist_copy.copy()
    distance[distance != 0] = 1
    return len(np.sum(np.unique(distance, axis=0), axis=1))
 
def auto_KMeans(data, K):
    minimal_claster_dist = (int(K * (K - 1) / 9)) + K
    distance = (data[:, 0][:, np.newaxis] - data[:, 0]) ** 2
    distance += (data[:, 1][:, np.newaxis] - data[:, 1]) ** 2
    distance = np.sqrt(distance)
    minimal_claster_dist = np.sort(distance.ravel())[minimal_claster_dist]
    minimal_claster_dist *= 0.72
    distance[distance <= minimal_claster_dist] = 0
    dist_copy = np.zeros(K * K).reshape(K, K)
    while True:
        for i in range(K):
            zero_dist = distance[distance [i, :] == 0]
            zero_dist = np.min(zero_dist, axis=0)
            dist_copy[i, :] = zero_dist
        if np.array_equal(dist_copy, distance):
            break
        distance = dist_copy.copy()
    distance[distance != 0] = 1
    return len(np.sum(np.unique(distance, axis=0), axis=1))

def swap_clusters(a, b, centroids, cluster_assignments):
    core = copy.deepcopy(centroids[a])
    centroids[a] = centroids[b]
    centroids[b] = core

    cluster_a_ind = np.argwhere(cluster_assignments == a)
    cluster_a_ind = np.reshape(cluster_a_ind, np.size(cluster_a_ind))

    cluster_b_ind = np.argwhere(cluster_assignments == b)
    cluster_b_ind = np.reshape(cluster_b_ind, np.size(cluster_b_ind))
    
    cluster_assignments = np.array(cluster_assignments)
    
    cluster_assignments[cluster_a_ind] = b*1.0
    cluster_assignments[cluster_b_ind] = a*1.0

    return centroids, cluster_assignments


def set_the_nearest_vectors(from_core, to_core, centroids, cluster_assignments, data, K):
    cluster_assignments = np.array(cluster_assignments)
    for i, cluster in enumerate(cluster_assignments):
        from_indexes = np.argwhere(cluster_assignments == from_core)
        from_vectors = np.take(data, np.reshape(from_indexes, np.size(from_indexes)), axis=0)

    for i, cluster in enumerate(cluster_assignments):
        to_indexes = np.argwhere(cluster_assignments == to_core)
        to_vectors = np.take(data, np.reshape(to_indexes, np.size(to_indexes)), axis=0)
    if (np.size(from_vectors, axis=0) >= K) and (np.size(to_vectors, axis=0) >= K):
        pairs = np.zeros((K, np.size(data[0]), 2))
    else:
        pairs = np.zeros((min(np.size(from_vectors, axis=0), np.size(to_vectors, axis=0)), np.size(data[0]), 2))

    for i in range(np.size(pairs, axis=0)):
        min_dist = euclide_norm(from_vectors[0]-to_vectors[0])
        nearest_vectors = np.array([from_vectors[0], to_vectors[0]])
        for to_vector in to_vectors:
            for from_vector in from_vectors:
                dist = euclide_norm(from_vector - to_vector)
                if dist < min_dist:
                    min_dist = dist
                    nearest_vectors[0] = from_vector
                    nearest_vectors[1] = to_vector

        pairs[i] = nearest_vectors
        from_vectors = np.delete(from_vectors, from_vectors.tolist().index(nearest_vectors[0].tolist()), axis=0)
        to_vectors = np.delete(to_vectors, to_vectors.tolist().index(nearest_vectors[1].tolist()), axis=0)
    return pairs


def euclide_norm(vector):
    enorm = 0
    for coof in vector:
        enorm += coof ** 2
    return enorm
  
    
def average_dist_of_nearest_vecs(nearest_pairs):
    aver_dist = 0
    count = 0
    for pair in nearest_pairs:
        aver_dist += euclide_norm(pair[0] - pair[1])
        count += 1
    aver_dist /= (count * 1.0)
    return aver_dist    
    
def sort_cores(centroids, cluster_assignments, data, K):
    sorted_core_index = np.arange(np.size(centroids, axis=0))
    near_pairs = set_the_nearest_vectors(0, 1, centroids=centroids, cluster_assignments=cluster_assignments, data=data, K=K)
    min_dist = average_dist_of_nearest_vecs(near_pairs)


    for i, core1 in enumerate(centroids[:-1]):
        for j, core2 in enumerate(centroids[i + 1:]):
            near_pairs = set_the_nearest_vectors(i, i + j + 1, centroids=centroids, cluster_assignments=cluster_assignments, data=data,K=K)
            dist = average_dist_of_nearest_vecs(near_pairs)
            if dist < min_dist:
                min_dist = dist
                first_index_core_pair = [i, j + i + 1]

    centroids, cluster_assignments = swap_clusters(0, first_index_core_pair[0], centroids, cluster_assignments)
    centroids, cluster_assignments = swap_clusters(1, first_index_core_pair[1], centroids, cluster_assignments)

    for i, core1 in enumerate(centroids[1:-1]):
        min_dist_pairs = set_the_nearest_vectors(i, i + 1, centroids=centroids, cluster_assignments=cluster_assignments, data=data,
                                           K=K)
        min_dist = average_dist_of_nearest_vecs(min_dist_pairs)
        min_dist_index = i + 1
        for j, core2 in enumerate(centroids[i + 1:]):
            near_pairs = set_the_nearest_vectors(i, j + i + 1, centroids=centroids, cluster_assignments=cluster_assignments, data=data,
                                           K=K)
            dist = average_dist_of_nearest_vecs(near_pairs)
            if dist < min_dist:
                min_dist = dist
                min_dist_index = j + i + 1
        centroids, cluster_assignments = swap_clusters(i + 1, min_dist_index, centroids, cluster_assignments)


    return centroids, cluster_assignments
    
    
    
    
