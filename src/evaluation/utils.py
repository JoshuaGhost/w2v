import os
from numpy import hstack, vstack
from numpy import array
from numpy import sqrt
from numpy.linalg import eigh
from gensim.models.word2vec import Word2Vec as w2v
from pickle import load

from multiprocessing import Pool

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

new_dim = 500
def read_wv(fname):
    vocab = []
    vecs = []
    for line in open(fname, 'r'):
        word, vec = map(eval, line.split(':'))
        vocab.append(word)
        vecs.append(vec)
    return vocab, vecs

def load_embeddings(folder, filename, extension, norm, arch):
    vocab = []
    vecs = []
    fnames = os.popen('ls '+folder+'/'+filename+"*"+extension+'|grep -v syn1neg|grep -v syn0').read().split()
    if arch == 'mapreduce':
        p = Pool(18)
        wvs = p.map(read_wv, fnames)
        vocab = [wv[0] for wv in wvs]
        vocab = reduce(list.__add__, vocab)
        vecs = [wv[1] for wv in wvs]
        vecs = vstack(array(vecs))
        if norm:
            for i in xrange(vecs.shape[0]):
                vecs[i,:] /= sqrt((vecs[i, :] ** 2).sum(-1))
        return vocab, vecs

    elif arch == 'local':
        lsresult = os
        ms = [w2v.load(fname) for fname in fnames]
        vocab = reduce(lambda x, y: x.intersection(y), [set(m.wv.vocab.keys()) for m in ms])
        vocab = list(vocab)
        if norm:
            for model in ms:
                model.init_sims()
            vecs = [[m.wv.syn0norm[m.wv.vocab[word].index] for word in vocab] for m in ms]
        else:
            vecs = [[m.wv.syn0[m.wv.vocab[word].index] for word in vocab] for m in ms]
        return vocab, hstack(vecs)

def dim_reduce(vecs, eigval_threshold, mean_corr):
    ms = None
    if mean_corr:
        vecs = vecs-vecs.mean(axis=0)
    cov = vecs.T.dot(vecs)
    evalues, bases = eigh(cov)
    evalues = sqrt(evalues)
    bases = bases[evalues>eigval_threshold]
    vecs = vecs.dot(bases.T)
    return vecs

