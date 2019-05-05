import math
import random
import matplotlib.pyplot as plt
import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD
comm_rank = comm.Get_rank()
comm_size = comm.Get_size()

def sort_by_values(list1, values):
    s = np.argsort(values)
    sorted_list = []
    for i in s:
        if i in list1:
            sorted_list.append(i)
    return sorted_list


def fast_non_dominated_sort(values):
    l = values.shape[0]
    values1 = values[:, 0].reshape(l, 1)
    values2 = values[:, 1].reshape(l, 1)
    front = [[]]
    rank = np.zeros(l)

    domi12 = (values1 > values1.T) * (values2 > values2.T)
    domi12 = domi12 + (values1 >= values1.T) * (values2 > values2.T)
    domi12 = domi12 + (values1 > values1.T) * (values2 >= values2.T)

    S = [np.argwhere(domi12[i]).flatten() for i in range(l)]  # 支配谁
    n = np.sum(domi12, 0)  # 被几个支配
    rank[n == 0] = 0
    front[0] = np.argwhere(n == 0).flatten().tolist()

    i = 0
    while (front[i] != []):
        Q = []
        for p in front[i]:
            for q in S[p]:
                n[q] = n[q] - 1
                if (n[q] == 0):
                    rank[q] = i + 1
                    if q not in Q:
                        Q.append(q)
        i = i + 1
        front.append(Q)

    del front[len(front) - 1]
    return front


# def crowding_distance(values, front):
#     values1 = values[:, 0]
#     values2 = values[:, 1]
#     distance = np.zeros(len(front))
#     sorted1 = sort_by_values(front, values[:, 0])
#     sorted2 = sort_by_values(front, values[:, 1])
#     distance[0] = np.inf
#     distance[len(front) - 1] = np.inf
#     for k in range(len(front) - 2):
#         distance[k + 1] = distance[k + 1] + (values1[sorted1[k + 2]] - values2[sorted1[k]]) / (max(values1) - min(values1))
#         distance[k + 1] = distance[k + 1] + (values1[sorted2[k + 2]] - values2[sorted2[k]]) / (max(values2) - min(values2))
#     return distance

def crowding_distance(values, front):
    min_max_diff = values.max(0) - values.min(0)
    l = len(front)
    distance = np.zeros(l)
    distance[[0, -1]] = np.inf
    values1f = values[front, 0]
    values2f = values[front, 1]
    srt1 = np.argsort(values1f)
    srt2 = np.argsort(values2f)
    if l > 2:
        a = (values1f[srt1[2:]] - values2f[srt1[:l - 2]]) / (min_max_diff[0])
        b = (values1f[srt2[2:]] - values2f[srt2[:l - 2]]) / (min_max_diff[1])
        distance[1:l - 1] = a + b
    return distance

def crossover(a, b):
    # crossover
    n=a.shape[0]
    rand = np.random.rand(n)
    c = (a - b).flatten() / 2
    c[rand > 0.5] = (a + b).flatten()[rand > 0.5] / 2
    # mutation
    mutation_prob = np.random.rand(n)
    c[mutation_prob < 1] = min_x + (max_x - min_x) * mutation_prob
    return c


def function(x):
    if comm_rank == 0:
        all_data = x
    all_data = comm.bcast(all_data if comm_rank == 0 else None, root=0) #发送广播
    local_data_offset = np.linspace(0, all_data.shape[0], comm_size + 1).astype('int')
    local_n=local_data_offset[comm_rank + 1]-local_data_offset[comm_rank]
    local_data = all_data[local_data_offset[comm_rank]:local_data_offset[comm_rank + 1]]

    local=np.hstack([-local_data ** 2, -(local_data - 2) ** 2])

    combine_data = comm.gather(local, root=0)
    if comm_rank == 0:
        data = np.vstack([i for i in combine_data])
    return data


pop_size = 20
max_gen = 921
min_x, max_x= [-55,55]
solution = min_x + (max_x - min_x) * np.random.rand(pop_size, 1)
for gen in range(100):
    if gen%10==0:
        print(gen/10)
    function_values = function(solution)

    solution2 = crossover(solution[np.random.permutation(pop_size)], solution[np.random.permutation(pop_size)])
    solution2 = np.vstack([solution, solution2.reshape([pop_size, 1])])

    function_values2 = function(solution2)
    non_dominated_sorted_solution2 = fast_non_dominated_sort(function_values2)
    crowding_distance_values2 = [crowding_distance(function_values2, i) for i in non_dominated_sorted_solution2]

    new_solution = []
    for i in range(len(non_dominated_sorted_solution2)):
        new_solution=new_solution+sorted(non_dominated_sorted_solution2[i],reverse=True)
        if (len(new_solution) >= pop_size):
            new_solution = new_solution[:pop_size]
            break
    solution = np.array([solution2[new_solution]]).reshape(len(new_solution), 1)

function1 = [i * -1 for i in function_values[:, 0]]
function2 = [j * -1 for j in function_values[:, 1]]
plt.xlabel('Function 1', fontsize=15)
plt.ylabel('Function 2', fontsize=15)
plt.scatter(function1, function2)
plt.show()