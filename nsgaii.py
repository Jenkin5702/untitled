import matplotlib.pyplot as plt
import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD
comm_rank = comm.Get_rank()
comm_size = comm.Get_size()
pop_size = 400
min_x, max_x = [-10, 10]
solution = np.zeros([pop_size, 1])


def find_dominate_relation(values):
    vals = comm.bcast(values if comm_rank == 0 else None, root=0)
    offset = np.linspace(0, vals.shape[0], comm_size + 1).astype('int')
    local_vals = vals[offset[comm_rank]:offset[comm_rank + 1]]

    local_values1 = local_vals[:, 0].reshape(local_vals.shape[0], 1)
    local_values2 = local_vals[:, 1].reshape(local_vals.shape[0], 1)

    values1 = vals[:, 0].reshape(vals.shape[0], 1)
    values2 = vals[:, 1].reshape(vals.shape[0], 1)

    dom = (local_values1 > values1.T) * (local_values2 > values2.T)
    dominant12 = np.vstack([i for i in comm.gather(dom, root=0)])
    return dominant12


def find_dominate_relation2(values):
    values1 = values[:, 0].reshape(values.shape[0], 1)
    values2 = values[:, 1].reshape(values.shape[0], 1)
    # Complexity O(m*n^2)
    dominant12 = (values1 > values1.T) * (values2 > values2.T)
    return dominant12


def fast_non_dominated_sort(values):
    dominate_relation = find_dominate_relation(values)
    # Complexity O(n)
    return np.argsort(np.sum(dominate_relation, 0))[:pop_size]


def f(x):
    # Complexity O(n*m)
    return np.hstack([-x ** 2, -(x - 2) ** 2])


for _ in range(1000):
    # Complexity O(n)
    c = min_x + (max_x - min_x) * np.random.rand(solution.shape[0], 1)
    solution2 = np.vstack([solution, c])
    solution = solution2[fast_non_dominated_sort(f(solution2))]

plt.scatter(-(f(solution)[:, 0]), -(f(solution)[:, 1]))
plt.show()
