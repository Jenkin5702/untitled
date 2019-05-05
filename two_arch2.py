import numpy as np
import time

'''参数设置'''
n = 100
c = 7
m = 3
pc = 1
pm = 0.1
bu = 1
bd = 0
p = 1 / m
eta_m = 15
eta_c = 15


def find_non_dominated(POP):
    for i in POP[:, range(c, c + m)]:
        POP = POP[np.any(POP[:, range(c, c + m)] <= i, 1)]
    return POP


def SBX(POP):
    r1 = np.random.rand(n)
    r2 = np.random.rand(n, c)
    r3 = np.random.rand(n, c)
    A = np.array([np.random.permutation(POP.shape[0]) for i in range(n)])[:, [0, 1]]
    POP1 = POP[A[:, 0], :][:, range(c)]
    POP2 = POP[A[:, 1], :][:, range(c)]
    [y1, y2] = np.sort([POP1, POP2], 0)
    alpha = 2.0 - (1 + 2 * np.min([y1 - bd, bu - y2], 0) / abs(POP1 - POP2)) ** -(eta_c + 1)
    alpha[r3 <= 1 / alpha] = (alpha * r3)[r3 <= 1 / alpha]
    alpha[r3 > 1 / alpha] = (1 / (2.0 - alpha * r3))[r3 > 1 / alpha]
    aa = 0.5 * ((y1 + y2) - alpha ** (1 / (eta_c + 1)) * (y2 - y1))
    bb = 0.5 * ((y1 + y2) + alpha ** (1 / (eta_c + 1)) * (y2 - y1))
    aa[aa < 0] = 0
    aa[aa > 1] = 1
    bb[bb < 0] = 0
    bb[bb > 1] = 1
    r = np.array([aa, bb])[np.random.permutation(2)]
    NPOP = np.zeros([2 * n, c])
    NPOP[2 * np.arange(n), :][r2 <= 0.5] = r[0][r2 <= 0.5]
    NPOP[2 * np.arange(n) + 1, :][r2 <= 0.5] = r[1][r2 <= 0.5]
    NPOP[2 * np.arange(n), :][r2 > 0.5] = POP1[r2 > 0.5]
    NPOP[2 * np.arange(n) + 1, :][r2 > 0.5] = POP2[r2 > 0.5]
    NPOP[2 * np.arange(n), :][r1 > pc] = 0
    NPOP[2 * np.arange(n) + 1, :][r1 > pc] = 0

    return NPOP


def SBXCD(CA, DA):
    NPOP = np.zeros([2 * n, c])
    r1 = np.random.rand(n)
    r2 = np.random.rand(n, c)
    r3 = np.random.rand(n, c)
    r4 = np.random.rand(n, c)
    Ac = np.array([np.random.permutation(CA.shape[0]) for i in range(n)])[:, [0, 1]]
    Ad = np.array([np.random.permutation(DA.shape[0]) for i in range(n)])[:, [0, 1]]
    [k1, k2] = Ac.T[[0, 1]]

    kind = np.all(CA[k1, :][:, range(c, c + m)] <= CA[k2, :][:, range(c, c + m)], 1)
    k = k2
    k[kind] = k1[kind]
    k[r1 > pc] = (k1 if np.random.rand() > 0.5 else k2)[r1 > pc]

    y = Ad.T[0]
    DAr = DA[y, :][:, range(c)]
    CAr = CA[k, :][:, range(c)]
    beta = 1 + 2 * np.min([np.min([DAr, CAr], 0), 1 - np.max([DAr, CAr], 0)], 0) / (
        np.max([DAr, CAr], 0) - np.min([DAr, CAr], 0))
    alpha = 2.0 - beta ** -(eta_c + 1)
    betaq = alpha
    betaq[r3 <= 1 / alpha] = ((alpha * r3) ** (1 / (eta_c + 1.0)))[r3 <= 1 / alpha]
    betaq[r3 > 1 / alpha] = ((1 / (2.0 - alpha * r3)) ** (1 / (eta_c + 1)))[r3 > 1 / alpha]
    b1 = np.min([DAr, CAr], 0) + np.max([DAr, CAr], 0)
    b2 = betaq * (np.max([DAr, CAr], 0) - np.min([DAr, CAr], 0))
    NPOP[np.arange(n) * 2, :][:, np.arange(c)] = 0.5 * (b1 - b2)
    NPOP[np.arange(n) * 2 + 1, :][:, np.arange(c)] = 0.5 * (b1 + b2)
    NPOP[NPOP > 1] = 1
    NPOP[NPOP < 0] = 0
    NPOP[np.arange(n) * 2, :][:, np.arange(c)][r2 > 0.5] = DAr[r2 > 0.5]
    NPOP[np.arange(n) * 2 + 1, :][:, np.arange(c)][r2 > 0.5] = CAr[r2 > 0.5]
    NPOP[np.arange(n) * 2, :][:, np.arange(c)][r4 > pc] = 0
    NPOP[np.arange(n) * 2 + 1, :][:, np.arange(c)][r4 > pc] = 0
    return NPOP


def compute_objectives(POP):
    g = np.sum((POP[:, 2:c] - 0.5) ** 2 - np.cos(20 * np.pi * (POP[:, 2:c] - 0.5)), 1)
    g = 100 * ((c - 2) * np.ones(POP.shape[0]) + g)
    o1 = POP[:, 0] * POP[:, 1] * (1 + g)
    o2 = POP[:, 0] * (1 - POP[:, 1]) * (1 + g)
    o3 = (1 - POP[:, 0]) * (1 + g)
    return np.vstack([o1, o2, o3]).T


def mutation(POP):
    randk = (np.random.rand(n) * POP.shape[0]).astype(np.int16)
    NPOP = POP[randk, :][:, range(c)]
    delta = np.min(np.array([(NPOP - bd) / (bu - bd), (bu - NPOP) / (bu - bd)]), 0)
    r2 = np.random.rand(n, c)
    indt = np.random.rand(n, c)
    yy = abs((2 * r2 + abs(1 - 2 * r2) * ((1 - delta) ** (eta_m + 1))) ** (1 / (eta_m + 1)) - 1) * (bu - bd)
    NPOP[NPOP <= bd] = (np.random.rand(n, c) * (bu - bd) + bd)[NPOP <= bd]
    NPOP[NPOP > bd] = (NPOP + yy)[NPOP > bd]
    NPOP[indt > pm] = POP[randk, :][:, range(c)][indt > pm]
    return NPOP


def updateDTA(CPOP, DA):
    J = []
    CPOPccm = CPOP[:, range(c, c + m)]
    for i in range(CPOP.shape[0]):
        DAccm = DA[:, range(c, c + m)]
        I = np.any(DAccm < CPOPccm[i], 1)
        j = np.argwhere(np.all(DAccm < CPOPccm[i], 1) + np.all(DAccm == CPOPccm[i], 1))
        if (j.shape[0] == 0):
            J = J + [i]
        else:
            I[j[0][0]:] = True
        DA = DA[I]
    DA = np.vstack([DA, CPOP[J]])
    if DA.shape[0] > n:
        I_selectDAlp = np.hstack([np.argmax(DA[:, range(c, c + m)], 0), np.argmin(DA[:, range(c, c + m)], 0)])
        I_selectDAlp = np.unique(I_selectDAlp)
        NPOP_selectDAlp = DA[I_selectDAlp, :]
        DA = np.delete(DA, I_selectDAlp, axis=0)
        while NPOP_selectDAlp.shape[0] < n and DA.shape[0] > 0:
            d = [np.sum((abs(NPOP_selectDAlp - DA[i])[:, range(c, m)]), 1).min() for i in range(DA.shape[0])]
            d = np.array(d)
            NPOP_selectDAlp = np.vstack([NPOP_selectDAlp, DA[np.argmax(d), :]])
            DA = np.delete(DA, np.argmax(d), axis=0)
            DA = np.delete(DA, np.argwhere(d < 0.001), axis=0)
        NDA = NPOP_selectDAlp
    else:
        NDA = DA
    return NDA


def updateCTA(CA, NPOP1):
    P = np.unique(np.vstack([CA, NPOP1]) if CA.shape[1] == NPOP1.shape[1] else NPOP1, axis=0)

    CPOP_Fitness = (P - np.min(P, 0) / (np.max(P, 0) - np.min(P, 0)))[:, range(c, c + m)]
    t2 = np.array([np.max(CPOP_Fitness - CPOP_Fitness[i], 1) for i in range(P.shape[0])])
    F = -np.sum(np.exp(-t2 / (np.max(abs(t2), 1) * 0.05)), 1)
    CPOP_t = ((P - np.min(P, 0)) / (np.max(P, 0) - np.min(P, 0)))[:, range(c, c + m)]
    C = np.array([np.max(np.max(abs(CPOP_t[:, range(m)] - CPOP_t[i, range(m)]), 1)) for i in range(P.shape[0])]).T

    while P.shape[0] > n:
        I = np.argmin(F)
        t1 = CPOP_t[I, range(m)]
        CPOP_t = np.delete(CPOP_t, I, axis=0)
        t2 = np.max(t1 - CPOP_t, 1)
        P = np.delete(P, I, axis=0)
        F = np.delete(F, I, axis=0)
        F = F + np.exp(0 - t2 / (C[I] * 0.05)).reshape(F.shape)
        C = np.delete(C, I, axis=0)
    NCA = P
    return NCA


start = time.perf_counter()
POP = np.random.rand(n, c)
obj = compute_objectives(POP)
POP = np.hstack([POP, obj])
DAG = find_non_dominated(POP)

CAG = np.zeros([0, 0])
for g in range(50):
    if CAG.shape[0] == 0:
        NPOP1 = SBX(POP)
        obj1 = compute_objectives(NPOP1)
        NPOP1 = np.hstack([NPOP1, obj1])
    else:
        NPOP1 = SBXCD(CAG, DAG)
        obj1 = compute_objectives(NPOP1)
        NPOP1 = np.hstack([NPOP1, obj1])

    if CAG.shape[0] == 0:
        NPOP2 = mutation(DAG)
        obj2 = compute_objectives(NPOP2)
        NPOP2 = np.hstack([NPOP2, obj2])
    else:
        NPOP2 = mutation(CAG)
        obj2 = compute_objectives(NPOP2)
        NPOP2 = np.hstack([NPOP2, obj2])
    NPOP = np.vstack([NPOP1, NPOP2])
    CPOP = find_non_dominated(NPOP)
    CAG = updateCTA(CAG, NPOP)
    DAG = updateDTA(CPOP, DAG)

print(DAG)

elapsed = (time.perf_counter() - start)
print("Time used:", elapsed)
