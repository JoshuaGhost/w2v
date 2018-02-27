from scipy.linalg import svd, block_diag, eigh
from scipy.sparse.csgraph import laplacian
import numpy as np


def dim_reduce(vecs, eigval_threshold, mean_corr):
    if mean_corr:
        vecs = vecs-vecs.mean(axis=0)
    cov = vecs.T.dot(vecs)
    evalues, bases = eigh(cov)
    evalues = sqrt(evalues)
    bases = bases[evalues>eigval_threshold]
    vecs = vecs.dot(bases.T)


def low_rank_repr(x):
    u, s, v = svd(x.T, full_matrices=False)
    mask = s > 1
    v = v[mask]
    s = s[mask]
    r = (v.T * (1 - s ** -2)).dot(v)
    return r


def low_rank_align(x, y, cxy, d=None, miu=0.8):
    nx, dy = x.shape
    ny, dx = y.shape
    assert cxy.shape == (nx, ny), \
        'Correspondence matrix must be shape num_samples_X X num_samples_Y.'
    c = np.fliplr(block_diag(np.fliplr(cxy), np.fliplr(cxy.T)))
    if d is None:
        d = min(dx, dy)
    rx = low_rank_repr(x)
    ry = low_rank_repr(y)
    r = block_diag(rx, ry)
    tmp = np.eye(r.shape[0]) - r
    m = tmp.T.dot(tmp)
    l = laplacian(c)
    eigen_prob = (1 - miu) * m + 2 * miu * l
    _, f = eigh(eigen_prob, eigvals=(1, d), overwrite_a=True)
    xembed = f[:nx]
    yembed = f[nx:]
    return xembed, yembed
