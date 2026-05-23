import os
import numpy as np
import scipy.io as sio

from update.fun.solver import solver
from update.fun.utils import constructMatrix


def main():
    # Caltech101-7, WebKB, Yale, MSRC-v5, Cora, Handwritten
    dataset = 'Caltech101-7'

    data_path = os.path.join('./dataset/', f'{dataset}.mat')
    mat_data = sio.loadmat(data_path)

    X = mat_data['X'][0]
    Y = mat_data['Y']
    for i in range(len(X)):
        X[i] = (X[i] - np.mean(X[i], axis=1, keepdims=True)) / np.std(X[i], axis=1, keepdims=True)

    num = X[0].shape[0]
    V = len(X)
    c = len(np.unique(Y))

    lambda_parm = 1024
    k = [4]
    alpha_parm = [100]
    beta_parm = [0.01]
    threshold = 1e-5

    init_neighbors = 8
    ACC_matrix = np.zeros((3, len(k), len(alpha_parm), len(beta_parm)))
    for i in range(len(k)):
        for j in range(len(alpha_parm)):
            for z in range(len(beta_parm)):
                A = constructMatrix(X, k[i], init_neighbors)
                S, obj, C, E, mu = solver(num, V, lambda_parm, alpha_parm[j], beta_parm[z], k[i], c, A, Y,
                                          threshold)


if __name__ == '__main__':
    main()

