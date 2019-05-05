import matplotlib.pyplot as plt
import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD
comm_rank = comm.Get_rank()
comm_size = comm.Get_size()
pop_size = 400
min_x, max_x = [-10, 10]
x = np.zeros([pop_size, 1])

ca = x
da = x


def find_dominate_relation(values1, values2):
    vals1 = comm.bcast(values1 if comm_rank == 0 else None, root=0)
    vals2 = comm.bcast(values2 if comm_rank == 0 else None, root=0)
    offset1 = np.linspace(0, vals1.shape[0], comm_size + 1).astype('int')
    offset2 = np.linspace(0, vals2.shape[0], comm_size + 1).astype('int')

    local_vals = vals1[offset1[comm_rank]:offset1[comm_rank + 1]]

    local_values1 = local_vals[:, 0].reshape(local_vals.shape[0], 1)
    local_values2 = local_vals[:, 1].reshape(local_vals.shape[0], 1)

    values1 = vals2[:, 0].reshape(vals2.shape[0], 1)
    values2 = vals2[:, 1].reshape(vals2.shape[0], 1)

    dom = (local_values1 > values1.T) * (local_values2 > values2.T)
    # Complexity O(4m*n^2/comm_size)
    dominant12 = np.vstack([i for i in comm.gather(dom, root=0)])
    return dominant12


def fast_non_dominated_sort(values):
    dominate_relation = find_dominate_relation(values, values)
    return np.argsort(np.sum(dominate_relation, 0))[:pop_size]


def f(x):
    return np.hstack([-x ** 2, -(x - 2) ** 2])


# def two_arch(x,ca,da):
#     dominate_relation11 = find_dominate_relation(f(x), f(ca))
#     dominate_relation12 = find_dominate_relation(f(ca), f(x))
#
#     dominate_relation21 = find_dominate_relation(f(x), f(da))
#     dominate_relation22 = find_dominate_relation(f(da), f(x))
#
#     flag = np.zeros(x.shape[0])
#     ca=ca[np.sum(dominate_relation12,1)==0]
#     da=da[np.sum(dominate_relation22,1)==0]
#
#     non_dominate = (np.sum(dominate_relation11, 1) + np.sum(dominate_relation21, 1)) == 0
#     with_dominate = (np.sum(dominate_relation12, 0) + np.sum(dominate_relation22, 0)) > 0
#
#     flag[non_dominate*with_dominate]=1
#     flag[~non_dominate]=-1
#
#     ca=np.vstack([ca,x[flag==1]])
#     da=np.vstack([da,x[flag==0]])
#     return [ca,da]
#
# def update_ca(ca):
#     sorted_ind=np.argsort(f(ca))
#     ca=ca[sorted_ind][:pop_size]


for ind in range(1000):
    # Complexity O(n)
    c = min_x + (max_x - min_x) * np.random.rand(x.shape[0], 1)
    x2 = np.vstack([x, c])
    x = x2[fast_non_dominated_sort(f(x2))]
    # [ca,da] = two_arch(x,ca,da)


plt.scatter(-(f(x)[:, 0]), -(f(x)[:, 1]))
plt.show()
