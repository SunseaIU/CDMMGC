import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import confusion_matrix, mutual_info_score, normalized_mutual_info_score, adjusted_rand_score
from scipy.stats import entropy
from scipy.special import comb


def best_map_py(L1, L2):
    L1 = L1.flatten()
    L2 = L2.flatten()

    labels = np.unique(np.concatenate((L1, L2)))
    n_labels = len(labels)

    G = confusion_matrix(L1, L2, labels=labels)

    row_ind, col_ind = linear_sum_assignment(-G)

    map_dict = {labels[col]: labels[row] for row, col in zip(row_ind, col_ind)}

    dummy_val = -1
    if n_labels > 0:
        dummy_val = np.min(labels) - 1 if np.issubdtype(labels.dtype, np.number) else "_unmapped_"

    newL2 = np.array([map_dict.get(val, dummy_val) for val in L2])
    return newL2


def compute_acc_py(Y, res):
    return np.mean(Y.flatten() == res.flatten())


def compute_purity_py(Y, predY):
    Y = Y.flatten()
    predY = predY.flatten()

    G = confusion_matrix(Y, predY)

    return np.sum(np.max(G, axis=0)) / len(Y)


def compute_nmi_matlab_py(L1, L2):
    L1 = L1.flatten()
    L2 = L2.flatten()
    n = len(L1)
    if n == 0:
        return 0.0

    mi_nats = mutual_info_score(L1, L2)
    mi_bits = mi_nats / np.log(2)

    _, counts_L1 = np.unique(L1, return_counts=True)
    _, counts_L2 = np.unique(L2, return_counts=True)

    h1_bits = entropy(counts_L1 / n, base=2)
    h2_bits = entropy(counts_L2 / n, base=2)

    max_h = max(h1_bits, h2_bits)
    if max_h == 0:
        return 1.0 if mi_bits == 0 else 0.0

    mi_hat = mi_bits / max_h

    return mi_hat


def compute_f_pair_based_py(T, H):
    T = T.flatten()
    H = H.flatten()
    N = len(T)

    if N < 2:
        return 1.0, 1.0, 1.0

    numT = 0
    numH = 0
    numI = 0

    for n in range(N):
        for m in range(n + 1, N):
            Tn = (T[n] == T[m])
            Hn = (H[n] == H[m])

            if Tn:
                numT += 1
            if Hn:
                numH += 1
            if Tn and Hn:
                numI += 1

    p = 1.0 if numH == 0 else numI / numH
    r = 1.0 if numT == 0 else numI / numT
    f = 0.0 if (p + r) == 0 else (2 * p * r) / (p + r)

    return f, p, r


def rand_index_to_ari_py(Y, predY):
    return adjusted_rand_score(Y.flatten(), predY.flatten())


def clusteringMeasure(Y, predY):
    Y_flat = Y.flatten()
    predY_flat = predY.flatten()
    n = len(Y_flat)

    if n == 0:
        return {'ACC': 0, 'NMI': 0, 'Purity': 0, 'ARI': 0, 'F_pair': 0, 'P_pair': 0, 'R_pair': 0}

    res = best_map_py(Y_flat, predY_flat)

    ACC = compute_acc_py(Y_flat, res)

    Purity = compute_purity_py(Y_flat, predY_flat)

    NMI_sklearn = normalized_mutual_info_score(Y_flat, predY_flat, average_method='arithmetic')

    ARI = rand_index_to_ari_py(Y_flat, predY_flat)

    F_pair, P_pair, R_pair = compute_f_pair_based_py(Y_flat, res)

    result = {
        'ACC': ACC,
        'NMI': NMI_sklearn,
        'Purity': Purity,
        'ARI': ARI,
        'Fscore': F_pair,
        'Precision': P_pair,
        'Recall': R_pair
    }

    return result
