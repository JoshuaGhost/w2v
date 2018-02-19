import os
from numpy import hstack, vstack
from numpy import sqrt
from numpy.linalg import eigh
from gensim.models.word2vec import Word2Vec
from functools import reduce

from multiprocessing import Pool

from scipy.linalg import block_diag, eigh, svd
from scipy.sparse.csgraph import laplacian

import numpy as np

from scalable_learning.extrinsic_evaluation.web.embedding import Embedding

new_dim = 500


def low_rank_repr(x):
    u, s, v = svd(x.T, full_matrices=False)
    mask = s > 1
    v = v[mask]
    s = s[mask]
    r = (v.T * (1-s**-2)).dot(v)
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
    tmp = np.eye(r.shape[0])-r
    m = tmp.T.dot(tmp)
    l = laplacian(c)
    eigen_prob = (1-miu)*m + 2*miu*l
    _, f = eigh(eigen_prob, eigvals=(1,d), overwrite_a=True)
    xembed = f[:nx]
    yembed = f[nx:]
    return xembed, yembed


def read_wv_hadoop(fname):
    vocab = []
    vecs = []
    for line in open(fname, 'r'):
        word, vec = map(eval, line.split(':'))
        vocab.append(word)
        vecs.append(vec)
    vecs = np.array(vecs)
    for i in range(vecs.shape[0]):
        vecs[i, :] /= sqrt((vecs[i, :]**2).sum(-1))
    return Embedding(dict(zip(vocab, vecs)))


def read_wv_csv(fname):
    vocab, vecs = [], []
    for line in open(fname, 'r'):
        if len(line.strip()) == 0:
            break
        word, vec = line.split(',', 1)
        vocab.append(word)
        vecs.append(eval(vec))
    vecs = np.array(vecs, dtype=float)
    for i in range(vecs.shape[0]):
        vecs[i, :] /= sqrt((vecs[i, :]**2).sum(-1))
    return Embedding(dict(zip(vocab, vecs)))


def load_embeddings(folder, filename, extension, norm, arch, selected=None):
    vecs = []
    fnames = os.popen('ls '+folder+'/'+filename+"*"+extension+'|grep -v syn1neg|grep -v syn0').read().split()
    if arch == 'mapreduce':
        p = Pool(18)
        ebs = p.map(read_wv_hadoop, fnames)
        vocab = [embedding.vocabulary for embedding in ebs]
        vecs = [embedding.vectors for embedding in ebs]
        vocab = reduce(lambda x, y: x + y, vocab)
        vecs = vstack(vecs)
        #   if norm:
        #    for i in xrange(vecs.shape[0]):
        #        vecs[i,:] /= sqrt((vecs[i, :] ** 2).sum(-1))
        return Embedding.from_dict(dict(zip(vocab, vecs)))
    elif arch == 'local':
        ms = [Word2Vec.load(fname) for fname in fnames]
        vocab = reduce(lambda x, y: x.intersection(y), [set(m.wv.vocab.keys()) for m in ms])
        vocab = list(vocab)
        if norm:
            for model in ms:
                model.init_sims()
            vecs = [[m.wv.syn0norm[m.wv.vocab[word].index] for word in vocab] for m in ms]
        else:
            vecs = [[m.wv.syn0[m.wv.vocab[word].index] for word in vocab] for m in ms]
        return vocab, hstack(vecs)
    elif arch == 'submodels':
        p = Pool(10)
        fnames = [folder+'/'+filename+str(i)+extension for i in selected]
        wvs = p.map(read_wv_hadoop, fnames, arch)
        vocab = wvs[0][0]
        vecs = [wv[1] for wv in wvs]
        vecs = hstack(np.array(vecs))
        if norm:
            for i in range(vecs.shape[0]):
                vecs[i, :] /= sqrt((vecs[i, :] ** 2).sum(-1))
        return vocab, vecs
    elif arch == 'csv':
        p = Pool(10)
        wvs = p.map(read_wv_csv, fnames)
        return wvs



