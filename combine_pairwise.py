# -*- coding: utf-8 -*-
"""
    This script combine different embeddings model according to their
    vectors.

    Usage: python combine_pairwise.py
                  <sub model dir>
                  <num of sub models>
                  <combination order>
                  <combination strategy>
                  <use lra?>
                  <use normed vector?>
                  <directory of merged file>

    For order it can be:
        seq     : sequential merge
        bin     : dicotomicall merge
        MPE	: everytime combine the two with minimum error w.r.t.
                          orthogonal Procrustes error

    For combination strategy it can be:
        vadd	: merge using average through each dimension
        lint	: merge using average then normalization
        PCA	: merge under hilbert space
"""
import logging
import os
import sys
import os.path
import numpy as np
import pickle

from gensim import matutils
from gensim.models.word2vec import Word2Vec
from time import time as ctime
from scipy.sparse import lil_matrix
from collections import deque
from glob import glob
from multiprocessing import Pool

from config import LOG_FILE
from lra import low_rank_align

program = os.path.basename(sys.argv[0])
logger = logging.getLogger(program)
logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
logging.root.setLevel(level=logging.INFO)
#logging.root.setLevel(level=logging.ERROR)

fvocab = 'vocab/vocab.txt'
with open(fvocab, 'r') as f:
    common_vocab = [word.lower()[:-1] for word in f.readlines()]
common_vocab = set(common_vocab)

flog = LOG_FILE


def procrustes_trans(X, Y):
     s = Y.T.dot(X)
     _, V = np.linalg.eig(s.T.dot(s))
     _, W = np.linalg.eig(s.dot(s.T))
     return W.dot(V.T)

def procrustes_err(X, Y):
    T = procrustes_trans(X, Y)
    Ystar = Y.dot(T)
    E = Ystar - X
    return np.trace(E.T.dot(E))
    
def combine(X, Y, strategy, uselra):
    def _vector_add(X, Y):
        return X+Y

    def _linear_trans(X, Y):
        T = procrustes_trans(X, Y)
        Ystar = Y.dot(T)
        return _vector_add(X, Ystar)

    def _pca_transbase(X, Y):
        dim = X.shape[1]
        V = np.hstack((X, Y))
        R = V.T.dot(V)
        _, evecs = np.linalg.eigh(R)
        evecs = evecs[:, -dim:]
        return V.dot(evecs)
    
    if uselra:
        X, Y = low_rank_align(X, Y, np.eye(X.shape[0]))
    if strategy == 'lint':
        return _linear_trans(X, Y)
    if strategy == 'vadd':
        return _vector_add(X, Y)
    if strategy == 'PCA':
        return _pca_transbase(X, Y)

def merge(embeddings, order, strategy, uselra):
    def merge_seq(embeddings):
        embeddings = deque(embeddings)
        while len(embeddings) > 1:
            X = embeddings.popleft()
            Y = embeddings.popleft()
            Z = combine(X, Y, strategy, uselra)
            embeddings.appendleft(Z)
        return embeddings.popleft()

    def merge_bin(embeddings):
        embeddings = deque(embeddings)
        while len(embeddings) > 1:
            X = embeddings.popleft()
            Y = embeddings.popleft()
            Z = combine(X, Y, strategy, lra)
            embeddings.append(Z)
        return embeddings.popleft()

    def merge_MPE(embeddings):
        namelist = [i for i in range(len(embeddings))]
        err = np.eye(len(embeddings)*2-1)
        for i in namelist[:-1]:
            for j in namelist[i+1:]:
                err[i][j] = procrustes_err(embeddings[i], embeddings[j])
        p = len(namelist)
        while True:
            idx_i = namelist[0]
            idx_j = namelist[1]
            for _, i in enumerate(namelist[:-1]):
                for j in namelist[_ + 1:]:
                    if err[i][j] < err[idx_i][idx_j]:
                        idx_i = i
                        idx_j = j
            Z = combine(embeddings[idx_i], embeddings[idx_j], strategy)
            embeddings.append(Z)
            namelist.remove(idx_i)
            namelist.remove(idx_j)
            if len(namelist) == 0:
                break
            namelist.append(p)
            for i in namelist[:-1]:
                err[i][p] = procrustes_err(Z, embeddings[i])
            p += 1
        return embeddings[-1]

    if order == 'seq':
        return merge_seq(embeddings)
    elif order == 'bin':
        return merge_bin(embeddings)
    elif order == 'MPE':
        return merge_MPE(embeddings)
    
def parse_argvs(argvs):
    if len(argvs) < 5:
        print globals()['__doc__'] % locals()
        sys.exit(1)

    (dfrags, nfrags, order,
     strategy, uselra, normed,
     dout) = argvs[1:8]
    
    config = [nfrags, order, strategy]

    nfrags = int(nfrags)
    uselra = (uselra == 'Y')
    normed = (normed == 'Y')
    flog = LOG_FILE

    config += ['lra'] if uselra else []
    config += ['normed'] if normed else []
    fout_name = '-'.join(['merged', dfrags.split('/')[-2]] + config)
    fout_name += '.pkl'
    return dfrags, nfrags, order, strategy, uselra, normed, dout, fout_name

def load_model(mname):
    return Word2Vec.load(mname)

def retrieve_vocab_as_set(model):
    return set(model.wv.vocab)

def normalize_model(model):
    model.init_sims()
    return model

def load_common_vecs(model):
    return np.array([model.wv[word] for word in common_vocab])


if __name__ == '__main__':
    logger.info("running %s" % ' '.join(sys.argv))

    dfrags, nfrags, order, strategy, uselra, normed, dout, fout_name = parse_argvs(sys.argv)
    
    etime = -ctime()

    namelist_frags = [dfrags+'article.txt.'+str(i)+'.txt.w2v'
           	      for i in range(nfrags)]
    #models = map(Word2Vec.load, namelist_frags)
    models = Pool(nfrags).map(load_model, namelist_frags)
    if normed:
        models = Pool(nfrags).map(normalize_model, models)

    vocabs = Pool(nfrags).map(retrieve_vocab_as_set, models)
    common_vocab = reduce(lambda x, y: x.intersection(y),
                          vocabs + [common_vocab])
    common_vocab = list(common_vocab)

    vecs = Pool(nfrags).map(load_common_vecs, models)
    
    merged_embeddings = merge(vecs, order=order, strategy=strategy, uselra=uselra)
   
    with open(os.path.join(dout, fout_name), 'w+') as fout:
        pickle.dump(dict(zip(common_vocab, merged_embeddings)), fout)

    etime += ctime()
    
    from time import localtime, strftime
    stime = strftime("%Y-%m-%d %H:%M:%S", localtime())
    with open(flog, 'a+') as ftime:
        ftime.write("%s, %s, %s, %f\n" % (stime, 'combine', os.path.join(dout, fout_name), etime))

