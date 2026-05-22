import numpy as np


def constructMatrix(X, orders, k):
    n = X[0].shape[0]
    A = {}

    for v in range(len(X)):
        D = L2_distance(X[v].T, X[v].T)
        idx = np.argsort(D, axis=1)
        W = np.zeros((n, n))

        for i in range(n):
            di = D[i, idx[i, 1:k + 2]]

            id = idx[i, 1:k + 2]

            W[i, id] = (di[k] - di) / (k * di[k] - np.sum(di[:k]) + np.finfo(float).eps)
        A[(v, 0)] = (W + W.T) / 2

        for i in range(1, orders):
            A[(v, i)] = np.matmul(A[(v, i - 1)], A[(v, 0)])
    return A


def L2_distance(a, b):
    if a.shape[0] == 1:
        a = np.vstack([a, np.zeros((1, a.shape[1]))])
        b = np.vstack([b, np.zeros((1, b.shape[1]))])

    aa = np.sum(a * a, axis=0)
    bb = np.sum(b * b, axis=0)
    ab = np.dot(a.T, b)
    # d[i, j] = aa[i] + bb[j] - 2 * ab[i, j]
    d = np.tile(aa[:, np.newaxis], (1, bb.shape[0])) + np.tile(bb[np.newaxis, :], (aa.shape[0], 1)) - 2 * ab

    d = np.real(d)  # get the real part

    d = np.maximum(d, 0)
    return d


def cal_eig(L, c=None, isMax=0, isSym=1):
    if c is None:
        c = L.shape[0]
        isMax = 1
        isSym = 1
    elif c > L.shape[0]:
        c = L.shape[0]
    if isSym == 1:
        L = np.maximum(L, L.T)
    d, v = np.linalg.eigh(L)

    if isMax == 0:
        idx = np.argsort(d)  # 由小到大
    else:
        idx = np.argsort(d)[::-1]
    idx1 = idx[:c]
    eigval = d[idx1]
    eigvec = v[:, idx1]
    eigval_full = d[idx]
    return eigval, eigvec, eigval_full


def EProjSimplex_new(v, k=1):
    ft = 1
    n = len(v)

    v0 = v - np.mean(v) + k / n
    vmin = np.min(v0)

    if vmin < 0:
        f = 1
        lambda_m = 0
        while abs(f) > 10 ** -10:
            v1 = v0 - lambda_m
            posidx = v1 > 0
            npos = np.sum(posidx)
            g = -npos
            f = np.sum(v1[posidx]) - k
            lambda_m = lambda_m - f / g
            ft = ft + 1
            if ft > 100:
                x = np.maximum(v1, 0)
                break
        x = np.maximum(v1, 0)
    else:
        x = v0

    return x, ft
