from scipy.linalg import eigh
from scipy import sqrt
import numpy as np


def pca(wvs, eigval_threshold, mean_corr):
    vocab = list(wvs[0].keys())
    vecs = np.hstack(np.array(wv[word] for word in vocab) for wv in wvs)
    if mean_corr:
        vecs = vecs-vecs.mean(axis=0)
    cov = vecs.T.dot(vecs)
    evalues, bases = eigh(cov)
    evalues = sqrt(evalues)
    bases = bases[evalues>eigval_threshold]
    vecs = vecs.dot(bases.T)
    return dict(zip(vocab, vecs))


def lra():
    pass


def concat(wvs):
    vocab = list(wvs[0].keys())
    vecs = np.hstack(np.array(wv[word] for word in vocab) for wv in wvs)
