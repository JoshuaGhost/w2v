# -*- coding: utf-8 -*-

from scipy.linalg import block_diag, eigh, svd
from scipy.sparse.csgraph import laplacian

import numpy as np

def low_rank_repr(X):
    U, S, V = svd(X.T, full_matrices=False)
    mask = S>1
    V = V[mask]
    S = S[mask]
    R = (V.T * (1-S**-2)).dot(V)
    return R
    

def low_rank_align(X, Y, Cxy, d=None, miu=0.8):
    nx, dy = X.shape
    ny, dx = Y.shape
    assert Cxy.shape==(nx, ny), \
            'Correspondence matrix must be shape num_samples_X X num_samples_Y.'
    C = np.fliplr(block_diag(np.fliplr(Cxy), np.fliplr(Cxy.T)))
    if d is None:
        d = min(dx, dy)
    Rx = low_rank_repr(X)
    Ry = low_rank_repr(Y)
    R = block_diag(Rx, Ry)
    tmp = np.eye(R.shape[0])-R
    M = tmp.T.dot(tmp)
    L = laplacian(C)
    eigen_prob = (1-miu)*M + 2*miu*L
    _, F = eigh(eigen_prob, eigvals=(1,d), overwrite_a=True)
    Xembed = F[:nx]
    Yembed = F[nx:]
    return Xembed, Yembed



