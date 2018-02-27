import numpy as np
from .utils import low_rank_align, dim_reduce


def pca(wvs, eigval_threshold, mean_corr):
    embedding = concat(wvs)
    embedding.vectors = dim_reduce(embedding.vectors, eigval_threshold, mean_corr)
    return embedding


def lra(wvs):
    embedding = wvs[0]
    vocab = wvs[0].vocabulary
    vec = wvs[0].vectors
    for i, e in enumerate(wvs[1:]):
        fx, fy = low_rank_align(embedding.vectors, e.vectors, np.ones((embedding.vectors.shape[0], e.vectors.shape[0])))
        embedding.vectors = (fx * i + fy) / (i + 1)
    return embedding


def concat(wvs):
    embedding = wvs[0]
    for e in wvs[1:]:
        embedding.concat(e)
    return embedding
