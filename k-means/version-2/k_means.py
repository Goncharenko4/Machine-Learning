import matplotlib.pyplot as plt
import random
import math
import numpy as np
import copy
from sklearn.datasets import make_blobs


def SetRandomLine(file_name):
    file = open(file_name, 'r')
    lines = file.read().splitlines()
    randomline = random.choice(lines)
    file.close()
    randomline = list(map(lambda x: float(x), randomline.split()))
    return randomline


def SetCores(file_name, number_of_clusters=3):
    cores = []
    for i in range(number_of_clusters):
        core = SetRandomLine(file_name)
        if not (core in cores):
            cores.append(core)
    cores = np.array(cores)
    return cores


def EuclideNorm(vector):
    enorm = 0
    for coof in vector:
        enorm += coof ** 2
    # enorm = math.sqrt(enorm)
    return enorm


def EuclideDistance(vector1, vector2):
    return EuclideNorm(vector1 - vector2)


def GetLines(file_name):
    file = open(file_name, 'r')
    lines = file.read().splitlines()
    for i, line in enumerate(lines, 0):
        lines[i] = list(map(float, line.split()))
    lines = np.array(lines)
    return lines


def MakeClusterList(file_name):
    cores = SetCores(file_name)
    cluster_list = []
    lines = GetLines(file_name)
    for core in cores:
        cluster = Cluster(core)
        cluster_list.append(cluster)
    return cluster_list


def VectorGenerator(file_name, number_of_vectors):
    file = open(file_name, 'w')
    for i in range(number_of_vectors):
        string = (" ".join(list(map(str, np.random.choice(100, 2).tolist()))))
        file.write(string + '\n')
    file.close()


def PutIntoFile(lines, file_name):
    file = open(file_name, 'w')
    for line in lines:
        string = (" ".join(list(map(str, line))))
        file.write(string + '\n')
    file.close()


def MakeDistanceCoreArray(core_list, cluster_list, lines):
    distance_core_array = []
    for j, core in enumerate(core_list):
        distance_array = []
        indexes = np.argwhere(cluster_list == j)
        its_vectors = np.take(lines, np.reshape(indexes, np.size(indexes)), axis=0)
        sort_vectors = np.array(sorted(its_vectors.tolist(), key=EuclideNorm))
        distance_array = np.zeros(len(sort_vectors) - 1)
        for i in range(len(sort_vectors) - 1):
            distance_array[i] = EuclideDistance(sort_vectors[i + 1], sort_vectors[i])
        distance_core_array.append(distance_array)
    return distance_core_array


def AverageDistance(distance_core_array):
    average_distance_array = []
    for distance_array in distance_core_array:
        aver_dist = 0
        count = 0
        for distance in distance_array:
            aver_dist += distance
            count += 1
        average_distance_array.append(aver_dist / (count * 1.0))
    return average_distance_array


def FakeClusterUpCorrect(average_distance_array, core_list, cluster_list, lines):
    min_dist = min(average_distance_array)
    min_dist_ind = average_distance_array.index(min_dist)
    max_dist = max(average_distance_array)
    max_dist_ind = average_distance_array.index(max_dist)
    aver_aver_dist = 0
    count = 0
    for aver_dist in average_distance_array:
        aver_aver_dist += aver_dist
        count += 1
    aver_aver_dist /= (count * 1.0)
    if max_dist > 2 * aver_aver_dist or min_dist == max_dist:
        a = cluster_list.tolist()
        new_core = lines[a.index(max_dist_ind)] if all(lines[a.index(max_dist_ind)] != core_list[max_dist_ind]) else \
        lines[len(a) - a[-1::-1].index(max_dist_ind) - 1]
        core_list = np.concatenate((core_list, [new_core]), axis=0)
    core_list, cluster_list = Clustering(old_core_list=core_list, lines=lines, cluster_list=cluster_list)

    return core_list, cluster_list


def Clustering(old_core_list, lines, cluster_list):
    core_list = copy.deepcopy(old_core_list)
    for i, line in enumerate(lines):
        min_distance = EuclideDistance(core_list[0], line)
        parent_core = 0
        for j, core in enumerate(core_list):
            ed = EuclideDistance(core, line)
            if ed < min_distance:
                min_distance = ed
                parent_core = j
        cluster_list[i] = parent_core

    core_list = np.zeros(shape=(len(core_list[:, 1]), 2), dtype=float)
    cluster_list_counter = np.zeros(shape=(len(core_list[:, 1])), dtype=int)
    for i, cluster in enumerate(cluster_list):
        core_list[cluster_list[i]] += lines[i]
        cluster_list_counter[cluster_list[i]] += 1
    for i, core in enumerate(core_list):
        core_list[i] = core / (cluster_list_counter[i] * 1.0)

    flag = 0
    for i, core in enumerate(core_list):
        dist = abs(core_list[i] - old_core_list[i])
        

        if (dist[0] >= 0.001 or dist[1] >= 0.001):
            flag = 1
    if flag == 1:
        core_list, cluster_list = Clustering(core_list, lines, cluster_list)
    return core_list, cluster_list


def SetNearestVectors(from_core, to_core, core_list, cluster_list, lines, amount_of_vectors):
    for i, cluster in enumerate(cluster_list):
        from_indexes = np.argwhere(cluster_list == from_core)
        from_vectors = np.take(lines, np.reshape(from_indexes, np.size(from_indexes)), axis=0)

    for i, cluster in enumerate(cluster_list):
        to_indexes = np.argwhere(cluster_list == to_core)
        to_vectors = np.take(lines, np.reshape(to_indexes, np.size(to_indexes)), axis=0)
    if (np.size(from_vectors, axis=0) >= amount_of_vectors) and (np.size(to_vectors, axis=0) >= amount_of_vectors):
        pairs = np.zeros((amount_of_vectors, np.size(lines[0]), 2))
    else:
        pairs = np.zeros((min(np.size(from_vectors, axis=0), np.size(to_vectors, axis=0)), np.size(lines[0]), 2))

    for i in range(np.size(pairs, axis=0)):
        min_dist = EuclideDistance(from_vectors[0], to_vectors[0])
        nearest_vectors = np.array([from_vectors[0], to_vectors[0]])
        for to_vector in to_vectors:
            for from_vector in from_vectors:
                dist = EuclideDistance(from_vector, to_vector)
                if dist < min_dist:
                    min_dist = dist
                    nearest_vectors[0] = from_vector
                    nearest_vectors[1] = to_vector

        pairs[i] = nearest_vectors
        from_vectors = np.delete(from_vectors, from_vectors.tolist().index(nearest_vectors[0].tolist()), axis=0)
        to_vectors = np.delete(to_vectors, to_vectors.tolist().index(nearest_vectors[1].tolist()), axis=0)
    return pairs


def AverageDistOfNearestVecs(nearest_pairs):
    aver_dist = 0
    count = 0
    for pair in nearest_pairs:
        aver_dist += EuclideDistance(pair[0], pair[1])
        count += 1
    aver_dist /= (count * 1.0)
    return aver_dist


def ClusterDownCorrect(core_list, lines, cluster_list, amount_of_vectors, optimal_average_distance):
    new_core_list = copy.deepcopy(core_list)
    new_cluster_list = copy.deepcopy(cluster_list)
    flag = False
    for i, core1 in enumerate(core_list[:-1]):
        for j, core2 in enumerate(core_list[i + 1:]):
            near_pairs = SetNearestVectors(from_core=i, to_core=j + i + 1, core_list=core_list,
                                           cluster_list=cluster_list, lines=lines, amount_of_vectors=amount_of_vectors)
            aver_dist = AverageDistOfNearestVecs(nearest_pairs=near_pairs)
            if aver_dist < optimal_average_distance:
                new_core_list = np.delete(new_core_list, new_core_list.tolist().index(core1.tolist()), axis=0)
                indexes = np.argwhere(new_cluster_list == i)
                indexes = np.reshape(indexes, np.size(indexes))
                new_cluster_list[indexes] = j + i + 1
                indexes = np.argwhere(new_cluster_list > i)
                indexes = np.reshape(indexes, np.size(indexes))
                new_cluster_list[indexes] -= 1
                # centers recounting
                new_core_list = np.zeros(shape=(len(new_core_list[:, 1]), 2), dtype=float)
                new_cluster_list_counter = np.zeros(shape=(len(new_core_list[:, 1])), dtype=int)
                for i, new_cluster in enumerate(new_cluster_list):
                    new_core_list[new_cluster_list[i]] += lines[i]
                    new_cluster_list_counter[new_cluster_list[i]] += 1
                for i, new_core in enumerate(new_core_list):
                    new_core_list[i] = new_core / (new_cluster_list_counter[i] * 1.0)
                new_core_list, new_cluster_list = ClusterDownCorrect(core_list=new_core_list, lines=lines,
                                                                     cluster_list=new_cluster_list,
                                                                     amount_of_vectors=amount_of_vectors,
                                                                     optimal_average_distance=optimal_average_distance)
                flag = True
                break
        if flag:
            break

    return new_core_list, new_cluster_list


def ClusterCorrect(core_list, lines, cluster_list, amount_of_vectors, optimal_average_distance):
    new_core_list = copy.deepcopy(core_list)
    new_cluster_list = copy.deepcopy(cluster_list)
    for i, core in enumerate(core_list):
        new_core_list = np.concatenate((new_core_list, [lines[(np.argwhere(cluster_list == i)[0, 0])]]), axis=0)
    new_core_list, new_cluster_list = Clustering(old_core_list=new_core_list, cluster_list=new_cluster_list,
                                                 lines=lines)
    new_core_list, new_cluster_list = ClusterDownCorrect(core_list=new_core_list, lines=lines,
                                                         cluster_list=new_cluster_list,
                                                         amount_of_vectors=amount_of_vectors,
                                                         optimal_average_distance=optimal_average_distance)

    return new_core_list, new_cluster_list


def Combinations(n, k):
    return math.factorial(n) / (math.factorial(k) * math.factorial(n - k) * 1.0)


def SwapClusters(a, b, core_list, cluster_list):
    core = copy.deepcopy(core_list[a])
    core_list[a] = core_list[b]
    core_list[b] = core

    cluster_a_ind = np.argwhere(cluster_list == a)
    cluster_a_ind = np.reshape(cluster_a_ind, np.size(cluster_a_ind))

    cluster_b_ind = np.argwhere(cluster_list == b)
    cluster_b_ind = np.reshape(cluster_b_ind, np.size(cluster_b_ind))

    cluster_list[cluster_a_ind] = b
    cluster_list[cluster_b_ind] = a

    return core_list, cluster_list


def SortCores(core_list, cluster_list, lines, amount_of_vectors):
    sorted_core_index = np.arange(np.size(core_list, axis=0))

    first_index_core_pair = [0, 1]
    near_pairs = SetNearestVectors(from_core=0, to_core=1, core_list=core_list, cluster_list=cluster_list, lines=lines,
                                   amount_of_vectors=amount_of_vectors)
    min_dist = AverageDistOfNearestVecs(near_pairs)

    for i, core1 in enumerate(core_list[:-1]):
        for j, core2 in enumerate(core_list[i + 1:]):
            near_pairs = SetNearestVectors(i, i + j + 1, core_list=core_list, cluster_list=cluster_list, lines=lines,
                                           amount_of_vectors=amount_of_vectors)
            dist = AverageDistOfNearestVecs(near_pairs)
            if dist < min_dist:
                min_dist = dist
                first_index_core_pair = [i, j + i + 1]
    core_list, cluster_list = SwapClusters(0, first_index_core_pair[0], core_list, cluster_list)
    core_list, cluster_list = SwapClusters(1, first_index_core_pair[1], core_list, cluster_list)
    for i, core1 in enumerate(core_list[1:-1]):
        min_dist_pairs = SetNearestVectors(i, i + 1, core_list=core_list, cluster_list=cluster_list, lines=lines,
                                           amount_of_vectors=amount_of_vectors)
        min_dist = AverageDistOfNearestVecs(min_dist_pairs)
        min_dist_index = i + 1
        for j, core2 in enumerate(core_list[i + 1:]):
            near_pairs = SetNearestVectors(i, j + i + 1, core_list=core_list, cluster_list=cluster_list, lines=lines,
                                           amount_of_vectors=amount_of_vectors)
            dist = AverageDistOfNearestVecs(near_pairs)
            if dist < min_dist:
                min_dist = dist
                min_dist_index = j + i + 1
        core_list, cluster_list = SwapClusters(i + 1, min_dist_index, core_list, cluster_list)

    return core_list, cluster_list


def TakeDistanceTable(core_list, cluster_list, lines):
    i = np.argsort(cluster_list)
    lines2 = lines[i, :]

    D = (lines2[:, 0][:, np.newaxis] - lines2[:, 0]) ** 2
    D += (lines2[:, 1][:, np.newaxis] - lines2[:, 1]) ** 2
    D = np.sqrt(D)
    plt.figure(figsize=(5, 4))
    plt.imshow(D, cmap='Purples_r', aspect='auto')
    plt.xticks([])
    plt.yticks([])
    plt.colorbar(orientation='vertical', pad=0.06);
    plt.show()


def TakePlot(core_list, cluster_list, lines):
    for j, core in enumerate(core_list):
        x = [ ]
        y = [ ]
        for i, line in enumerate(lines):
            if (cluster_list[i] == j):
                x.append(line[0])
                y.append(line[1])
        plt.scatter(x, y)
    for core in core_list:
        plt.scatter(core[0], core[1], c='black')
    plt.show()