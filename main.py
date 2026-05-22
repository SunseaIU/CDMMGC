import os
import numpy as np
import scipy.io as sio
from fun.solver import solver
from fun.matrix import clusteringMeasure
from scipy.sparse import csgraph
from fun.utils import constructMatrix
import time


# method : \min_{S,F,C_k^v}\sum_{v=1}^V\sum_{k=1}^K \mu_k^v\|S-C_k^v\|_F^2+2+\alpha\|S\|_F^2+\lambda Tr(F^TL_SF)+\beta\sum_{k=1}^K\sum_{i,j=1}^Vw_{ij}\mu_k^i\mu_k^jTr((A_k^i-C_k^i)(A_k^j-C_k^j)^T) \\
# \\ s.t. \quad s^T1=1,S\ge0,F\in R^{n\times c},F^TF=I,A_k^v\ge C_k^v \ge 0,rank(L_s) = n-c
def main():
    # Caltech101-7, WebKB, Yale, MSRC-v5, Cora, Handwritten
    dataset = 'Caltech101-7'

    log_path = f'log/'
    if not os.path.exists(log_path):
        os.makedirs(log_path)

    data_path = os.path.join('./dataset/', f'{dataset}.mat')
    mat_data = sio.loadmat(data_path)

    if dataset == 'Caltech101-7' or dataset == 'HW' or dataset == 'MSRC_v1' or dataset == 'ORL' or dataset == 'Caltech101-20':
        X = mat_data['X'][0]
        Y = mat_data['Y']
        for i in range(len(X)):
            X[i] = (X[i] - np.mean(X[i], axis=1, keepdims=True)) / np.std(X[i], axis=1, keepdims=True)
    else:
        X = mat_data['X']
        Y = mat_data['y']
        temp_X = []
        for i in range(len(X)):
            temp_X.append((X[i][0] - np.mean(X[i][0], axis=1, keepdims=True)) / (
                    np.std(X[i][0], axis=1, keepdims=True) + np.finfo(float).eps))
        X = temp_X  # num of samples



    num = X[0].shape[0]
    # num of views
    V = len(X)
    # num of clusters
    c = len(np.unique(Y))

    lambda_parm = 1024
    # lambda_parm = 1e3
    # k = [2, 3, 4, 5, 6]
    # alpha_parm = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
    # beta_parm = [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
    k = [4]
    alpha_parm = [1000]

    beta_parm = [10]

    init_neighbors = 8
    ACC_matrix = np.zeros((3, len(k), len(alpha_parm), len(beta_parm)))
    for i in range(len(k)):  # 测试不同的阶数对应的性能
        for j in range(len(alpha_parm)):
            for z in range(len(beta_parm)):
                A = constructMatrix(X, k[i], init_neighbors)
                S, obj, C, E, mu = solver(num, V, lambda_parm, alpha_parm[j], beta_parm[z], k[i], c, A, Y)
                S[S < 1e-5] = 0

                # np.save(f'../vision/{dataset}/A.npy', A)
                np.save(f'./S/{dataset}_S.npy', S)
                # np.save(f'../vision/{dataset}/C.npy', C)
                # np.save(f'../vision/{dataset}/E.npy', E)

                # Y_col = Y.reshape(-1, 1)
                # ground_true_bool = (Y_col == Y.reshape(1, num))
                # ground_true = ground_true_bool.astype(int)
                # np.save(f'../vision/{dataset}/ground_true.npy', ground_true)

                # 获取结果
                n_components, labels = csgraph.connected_components(csgraph=S, directed=False)
                final = labels
                # 评估聚类结果
                result = clusteringMeasure(Y, final)
                ACC_matrix[0, i, j, z] = result['ACC']
                ACC_matrix[1, i, j, z] = result['Purity']
                ACC_matrix[2, i, j, z] = result['NMI']
                print('+++++++++++++++++++++++++++++++++++++++++++')
                print(
                    f"  ACC:{result['ACC']:.6f}  NMI:{result['NMI']:.6f}  Purity:{result['Purity']:.6f}  ARI:{result['ARI']:.6f}")
                print(
                    f"  Precision:{result['Precision']:.6f}  Recall:{result['Recall']:.6f}  Fscore:{result['Fscore']:.6f}")
                print('+++++++++++++++++++++++++++++++++++++++++++')

                with open(os.path.join(log_path, f'{dataset}.txt'), 'a', encoding='utf-8') as f:
                    f.write("============================\n")
                    f.write(f"训练时间: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write('parameter(lambda_parm, alpha_parm, beta_parm, init_neighbors, orders)\n')
                    f.write(f'lambda_parm:{lambda_parm}\n')
                    f.write(f'alpha_parm:{alpha_parm[j]}\n')
                    f.write(f'beta_parm:{beta_parm[z]}\n')
                    f.write(f"init_neighbors:{init_neighbors}\n")
                    f.write(f'init_orders:{k[i]}\n')
                    # f.write(f'dataset: {dataset} ACC:{result["ACC"]} Purity:{result["Purity"]} NMI:{result["NMI"]}\n')
                    f.write(
                        f"  ACC:{result['ACC']:.6f}  NMI:{result['NMI']:.6f}  Purity:{result['Purity']:.6f}  ARI:{result['ARI']:.6f}\n")
                    f.write(
                        f"  Precision:{result['Precision']:.6f}  Recall:{result['Recall']:.6f}  Fscore:{result['Fscore']:.6f}\n")
    # print(ACC_matrix)
    # print(np.max(ACC_matrix[0]))
    np.save(f'./log/{dataset}.npy', ACC_matrix)


if __name__ == '__main__':
    main()
