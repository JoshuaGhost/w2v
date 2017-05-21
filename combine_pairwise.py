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
                  <name of merged file>

    For order it can be:
        sequential      : sequential merge
        dichoto         : dicotomicall merge
        min_procrustes_err  : everytime combine the two with minimum error w.r.t.
                          orthogonal Procrustes error

    For combination strategy it can be:
        vector_addition : merge using average through each dimension
        linear_transform: merge using average then normalization
        pca_transbase   : merge under hilbert space
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

from config import LOG_FILE
from lra import low_rank_align

def load_vocab(vocab_file):
    with open(vocab_file, 'r') as f:
        vocab = [word.lower()[:-1] for word in f.readlines()]
    return vocab
        
def vocab_intersect(v1, v2):
    return v1.intersection(v2)

def load_common_vectors(common_vocab, normed):
    def _load_vectors(mname):
        model = Word2Vec.load(mname)
        if normed:
            model.init_sims()
        vectors = [model.wv[word] for word in common_vocab]
        return np.array(vectors)
    return _load_vectors

def vocab_intersect(v1, v2):
    return v1.intersection(v2)

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
    
def combine(X, Y, strategy, lra=False):
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
        _, evecs = np.linalg.eig(R)
        evecs = evecs[:, :dim]
        return V.dot(evecs)
    
    if lra:
        X, Y = low_rank_align(X, Y, np.eye(X.shape[0]))
    if strategy == 'linear_transform':
        return _linear_trans(X, Y)
    if strategy == 'vector_addition':
        return _vector_add(X, Y)
    if strategy == 'pca_transbase':
        return _pca_transbase(X, Y)

def merge(embeddings, order='sequential', strategy='avg', lra=False):
    def merge_seq(embeddings):
        embeddings = deque(embeddings)
        while len(embeddings) > 1:
            X = embeddings.popleft()
            Y = embeddings.popleft()
            Z = combine(X, Y, strategy, lra)
            embeddings.appendleft(Z)
        return embeddings.popleft()

    def merge_dichoto(embeddings):
        embeddings = deque(embeddings)
        while len(embeddings) > 1:
            X = embeddings.popleft()
            Y = embeddings.popleft()
            Z = combine(X, Y, strategy, lra)
            embeddings.append(Z)
        return embeddings.popleft()

    def merge_min_procrustes_err(embeddings):
        namelist = [i for i in range(len(embeddings))]
        err = np.eye(len(embeddings)) * 3276899999.0
        for i in namelist[:-1]:
            for j in namelist[i+1:]:
                err[i][j] = procrustes_err(embeddings[i], embeddings[j])
                err[j][i] = err[i][j]
        p = len(namelist)
        while True:
            idx_i = namelist[0]
            idx_j = namelist[0]
            for _, i in enumerate(namelist[:-1]):
                for j in namelist[_ + 1:]:
                    if err[i][j] < err[idx_i][idx_j]:
                        idx_i = i
                        idx_j = j
            Z = combine(embeddings[idx_i], embeddings[idx_j], strategy)
            embeddings.append(Z)
            namelist.remove(idx_i)
            namelist.remove(idy_j)
            if len(namelist) == 0:
                break
            namelist.append(p)
            err = np.vstack((err, np.zeros((1, err.shape[1]))))
            err = np.hstack((err, np.zeros((err.shape[0], 1))))
            err[-1][-1] = 32768.0
            for i in namelist[:-1]:
                err[i][p] = procrustes_err(Z, embeddings[i])
                err[p][i] = err[i][p]
            p += 1
        return embeddings[-1]

    if order == 'sequential':
        return merge_seq(embeddings)
    elif order == 'dichoto':
        return merge_dichoto(embeddings)
    elif order == 'min_procrustes_err':
        return merge_min_procrustes_err(embeddings)
    

if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])
    global logger
    logger = logging.getLogger(program)
    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level=logging.INFO)
    logger.info("running %s" % ' '.join(sys.argv))

    if len(sys.argv) < 5:
        print globals()['__doc__'] % locals()
        sys.exit(1)

    (sub_models_dir, num_sub_models, order,
     strategy, lra, normed,
     output_folder) = sys.argv[1:8]
    
    config = [num_sub_models, order, strategy]

    num_sub_models = int(num_sub_models)
    lra = (lra == 'Y')
    normed = (normed == 'Y')
    log_file = LOG_FILE
    benchmark_vocab_dir = 'vocab/vocab.txt'

    config += ['lra'] if lra else []
    config += ['normed'] if normed else []
    output_file = '-'.join(['merge'] + config)
    output_file += '.pkl'

    ftime = open(log_file, 'a+')
    ftime.write('time of alignment and combination with order:'
                ' %s and strategy: %s\n' % (order, strategy))
    ftime.write('sub-models under %s\n' % sub_models_dir)
    ftime.write('combined models saved as %s\n\n' % os.path.join(output_folder, output_file))

    time_elapsed = -ctime()

    model_name_list = [sub_models_dir+'article.txt.'+str(i)+'.txt.w2v'
           for i in range(num_sub_models)]
    vocabs = [set(Word2Vec.load(mname).wv.vocab) for mname in model_name_list]
    init_vocab = set(load_vocab(benchmark_vocab_dir))
    common_vocab = reduce(lambda x, y: x.intersection(y),
                          vocabs,
                          init_vocab)
    common_vocab = list(common_vocab)
    embeddings = map(load_common_vectors(common_vocab, normed), model_name_list)
    
    merged_embeddings = merge(embeddings, order=order, strategy=strategy, lra=lra)
   
    with open(os.path.join(output_folder, output_file), 'w+') as ofile:
        pickle.dump(dict(zip(common_vocab, merged_embeddings)), ofile)

    time_elapsed += ctime()
    
    ftime.write('combination time: %f sec\n\n' % time_elapsed)

    from time import localtime, strftime
    times = strftime("%Y-%m-%d %H:%M:%S", localtime())

    ftime.write("full-stack of experiment finished at %s\n\n" % times)
    ftime.close()

