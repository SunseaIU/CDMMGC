import numpy as np
from .utils import cal_eig, EProjSimplex_new
from .matrix import clusteringMeasure
from scipy.sparse import csgraph
import random


def solver(num, V, lambda_param, alpha_param, beta_param, K, c, A, Y, threshold):
    rd = np.random.RandomState(888)
    SS = 0
    for v in range(V):
        SS = SS + A[(v, 0)]
    SS = (SS + SS.T) / 2

    C = {}
    for v in range(V):
        for k in range(K):
            # C[(v, k)] = rd.uniform(0, 0.3, (num, num))
            C[(v, k)] = np.zeros((num, num))
            # C[(v, k)] = A[(v, k)]

    mu = np.zeros((V, K))
    for v in range(V):
        for k in range(K):
            # mu[v, k] = 1 / (2 * np.sqrt(np.sum((SS - C[(v, k)]) ** 2)) + np.finfo(float).eps)
            mu[v, k] = random.uniform(0.0, 0.60)

    a1 = 1
    a2 = 1
    W = np.full((V, V), a1)
    W[np.diag_indices_from(W)] = a2

    L = np.diag(np.sum(SS, axis=1)) - SS + np.eye(num) * np.finfo(float).eps
    eigvals, eigvecs, eigval_full = cal_eig(L, c)
    F = eigvecs[:, :c]
    F = F / np.sqrt(np.sum(F ** 2, axis=1) + np.finfo(float).eps)[:, np.newaxis]

    iter_max = 25
    obj = np.zeros(iter_max)

    S_old = None

    for t in range(iter_max):

        # update S
        distance = np.zeros((num, num))
        for i in range(num):
            for j in range(num):
                distance[i, j] = np.sum((F[i] - F[j]) ** 2)
        D = lambda_param * distance
        B = np.zeros((num, num))
        for v in range(V):
            for k in range(K):
                B = B + mu[v, k] * C[(v, k)]
        M = D - 2 * B

        H = np.sum(mu) + np.finfo(float).eps

        S = np.zeros((num, num))
        for i in range(num):
            v_i = -M[i, :] / (2 * H + alpha_param)
            tempS, _ = EProjSimplex_new(v_i)
            S[i, :] = tempS

        # update C
        C = update_C(A, S, mu, W, V, K, beta_param)

        # update mu
        for v in range(V):
            for k in range(K):
                mu[v, k] = 1 / (2 * np.sqrt(np.sum((S - C[(v, k)]) ** 2)) + np.finfo(float).eps)
        # print('===============')
        # print(mu)
        # print('===============')
        # update F
        S = (S + S.T) / 2
        L = np.diag(np.sum(S, axis=1)) - S + np.eye(num) * np.finfo(float).eps
        F_old = F
        eigvals, F, eigval_full = cal_eig(L, c, 0)

        # ==================== 计算目标函数值 ====================
        # Term 1: Σ Σ μ_k^v ||S - C_k^v||_F^2
        # 由于 mu 的定义，该项简化为 0.5 * Σ Σ ||S - C_k^v||_F
        term1 = 0
        for v_i in range(V):
            for k_i in range(K):
                # np.linalg.norm(..., 'fro') 计算 Frobenius 范数
                term1 += 0.5 * np.linalg.norm(S - C[(v_i, k_i)], 'fro')

        # Term 2: 2 * λ * Tr(F^T * Ls * F)
        # Tr(F^T * Ls * F) 等于 Ls 的前 c 个最小特征值之和
        fn1 = np.sum(eigval_full[:c])
        term2 = 2 * lambda_param * fn1

        # Term 3: Σ_k Σ_{i,j} w_ij * μ_k^i * μ_k^j * Tr((A_k^i - C_k^i)(A_k^j - C_k^j)^T)
        term3 = 0
        for k_i in range(K):
            mu_k = mu[:, k_i]
            mu_k_outer = np.outer(mu_k, mu_k)
            diversity_coeff_part = W * mu_k_outer

            diff_matrices = [A[(v_i, k_i)] - C[(v_i, k_i)] for v_i in range(V)]

            for i in range(V):
                for j in range(V):
                    trace_val = np.sum(diff_matrices[i] * diff_matrices[j])  # Frobenius 内积
                    term3 += diversity_coeff_part[i, j] * trace_val

        obj[t] = term1 + term2 + term3

        fn2 = np.sum(eigval_full[:c + 1])

        if fn1 > threshold:  # 簇数少于 c
            lambda_param = lambda_param * 2
        elif fn2 < threshold:  # 簇数多于 c
            lambda_param = lambda_param / 2
            F = F_old
        else:
            break
        S[S < 1e-5] = 0

        n_components, labels = csgraph.connected_components(csgraph=S, directed=False)
        final = labels
        result = clusteringMeasure(Y, final)

        print(
            f"  ACC:{result['ACC']:.6f}  Purity:{result['Purity']:.6f}  NMI:{result['NMI']:.6f}  ARI:{result['ARI']:.6f}")
        print(
            f"  Precision:{result['Precision']:.6f}  Recall:{result['Recall']:.6f}  Fscore:{result['Fscore']:.6f}")

    E = {}
    for v in range(V):
        for k in range(K):
            E[(v, k)] = A[(v, k)] - C[(v, k)]
    return S, obj, C, E, mu


def update_C(A, S, mu, W, V, K, beta_param):
    num = S.shape[0]

    C = {}
    for k in range(K):
        h_k = mu[:, k]
        mu_k = mu[:, k]

        mu_k_outer = np.outer(mu_k, mu_k)
        diversity_coeff_part = beta_param * W * mu_k_outer
        M_k = np.diag(mu_k) + diversity_coeff_part

        # P_k^v = mu_vk * S + Σ_j (w_vj * μ_kv * μ_kj * A_k^j)
        P_k_1 = mu_k[:, np.newaxis, np.newaxis] * S  # (V, n, n)
        temp_A = []
        for v in range(V):
            temp_A.append(A[(v, k)])  # (V, n, n)
        temp_A_reshaped = np.reshape(temp_A, (V, -1))  # (V, n*n)
        P_k_2_reshaped = diversity_coeff_part @ temp_A_reshaped
        P_k_2 = np.reshape(P_k_2_reshaped, (V, num, num))
        P_k = P_k_1 + beta_param * P_k_2

        P_k_reshaped = np.reshape(P_k, (V, -1))
        try:
            C_k_solved_reshaped = np.linalg.solve(M_k, P_k_reshaped)
            C_k_solved = C_k_solved_reshaped.reshape(V, num, num)
        except np.linalg.LinAlgError:
            M_k_pinv = np.linalg.pinv(M_k)
            C_k_solved_reshaped = M_k_pinv @ P_k_reshaped
            C_k_solved = C_k_solved_reshaped.reshape(V, num, num)
        C_k_projected = np.clip(C_k_solved, 0, temp_A)

        for v in range(V):
            C[(v, k)] = C_k_projected[v]

    return C
