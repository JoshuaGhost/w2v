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

from lra import low_rank_align
from tsne_smp import tsne

program = os.path.basename(sys.argv[0])
logger = logging.getLogger(program)
logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
logging.root.setLevel(level=logging.DEBUG)
#logging.root.setLevel(level=logging.ERROR)

fvocab = '../../vocab/vocab.txt'
with open(fvocab, 'r') as f:
    common_vocab = [word.lower()[:-1] for word in f.readlines()]
common_vocab = set(common_vocab)


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
    if strategy == 'tsne':
        Z = np.hstack((X,Y))
        return tsne(Z, no_dims=500, initial_dims=1000)


def merge(embeddings, order, strategy, uselra):
    def _merge_seq(embeddings):
        embeddings = deque(embeddings)
        while len(embeddings) > 1:
            X = embeddings.popleft()
            Y = embeddings.popleft()
            Z = combine(X, Y, strategy, uselra)
            embeddings.appendleft(Z)
        return embeddings.popleft()

    def _merge_bin(embeddings):
        embeddings = deque(embeddings)
        while len(embeddings) > 1:
            X = embeddings.popleft()
            Y = embeddings.popleft()
            Z = combine(X, Y, strategy, uselra)
            embeddings.append(Z)
        return embeddings.popleft()

    def _merge_MPE(embeddings):
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
            Z = combine(embeddings[idx_i], embeddings[idx_j], strategy, uselra)
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
        return _merge_seq(embeddings)
    elif order == 'bin':
        return _merge_bin(embeddings)
    elif order == 'MPE':
        return _merge_MPE(embeddings)


def parse_argvs(argvs):
    import argparse

    parser=argparse.ArgumentParser(description='Process some arguments.')

    parser.add_argument('-s', '--source', dest='dfrags',\
                        default="../../temp/models/two_parts_bootstrap/",\
                        help='source directory that contains model fragments entail to merge')
    parser.add_argument('-n', '--nfrags', type=int, default=2,\
                        help='number of fragments require to merge')
    parser.add_argument('--order', default='seq',\
                        help='order of merging, can be \
                                seq (for sequence), \
                                bin (for binary) and \
                                MPE (for min Procrustes error)')
    parser.add_argument('-l', '--lra', type=bool, dest='uselra', default=False,\
                        help='whether use lra to arrange')
    parser.add_argument('--norm', type=bool, dest='normed', default=False,\
                        help='whether normalize the models')
    parser.add_argument('--strategy', default='tsne',\
                        help='strategy to use when pairwise combining, can be \
                                vadd (for vector add), \
                                lint (for linear transformation) and \
                                PCA')
    parser.add_argument('-d', '--destination', dest='dout', default="../../output/models/two/",\
                        help='destination directory to save merged model')
    parser.add_argument('-o', '--output', dest='fout_name', default="tsne_two_part.pkl",\
                        help='name of output file')

    ns = parser.parse_args(argvs)
    mdivide = ns.dfrags.split('/')[-2]
    tmp_dir = 'temp/' + (mdivide+'/') + ('normed/' if ns.normed else 'origin/')
 
    return (ns.dfrags, mdivide, ns.nfrags,
            ns.order, ns.strategy, ns.uselra,
            ns.normed, ns.dout, ns.fout_name,
            tmp_dir)


def load_model(mname):
    return Word2Vec.load(mname)

def retrieve_vocab_as_set(model):
    return set(model.wv.vocab)

def load_common_vecs(vmpair):
    return np.array([vmpair[1].wv[word] for word in vmpair[0]])


def retrieve_vecs(dfrags, nfrags, tmp_dir):
    vocab = common_vocab
    if os.path.isfile(tmp_dir+'vocab.pkl') and os.path.isfile(tmp_dir+'vecs.pkl'):
        logger.info('vocab and vecs already exist under %s' % tmp_dir)

        with open(tmp_dir+'vocab.pkl', 'r') as fvocab:
            vocab = pickle.load(fvocab)
        with open(tmp_dir+'vecs.pkl', 'r') as fvecs:
            vecs = pickle.load(fvecs)
    else:
        logger.info('{generating,dumping} {vocab,vecs} files under %s' % tmp_dir)

        logger.debug('loading models...')
        models = Pool(nfrags).map(load_model, [dfrags+'article.txt.'+str(i)+'.txt.w2v' for i in range(nfrags)])
        logger.debug('models loaded')

        if normed:
            logger.debug('normalizing models...')
            for model in models:
                model.init_sims()
            logger.debug('models normalization complete')

        logger.debug('extracting common vocabs...')
        vocabs = Pool(nfrags).map(retrieve_vocab_as_set, models)+[vocab]
        vocab = list(reduce(lambda x, y: x.intersection(y), vocabs))
        with open(tmp_dir+'vocab.pkl', 'w+') as fvocab:
            pickle.dump(vocab, fvocab)
	logger.debug('vocab extraction complete')

        logger.debug('extracting corresponding vectors...')
        vecs = Pool(nfrags).map(load_common_vecs, zip([vocab for i in range(len(models))], models))
        with open(tmp_dir+'vecs.pkl', 'w+') as fvecs:
            pickle.dump(vecs, fvecs)
        logger.debug('vectors extraction complete')

    return vocab, vecs


if __name__ == '__main__':
    logger.info("running %s" % ' '.join(sys.argv))

    (dfrags, mdivide, nfrags,
     order, strategy, uselra,
     normed, dout, fout_name,
     tmp_dir) = parse_argvs(sys.argv[1:])
    
    logger.debug(tmp_dir)
    logger.info('retrieving common vocabulary and vectors...')
    common_vocab, vecs = retrieve_vecs(dfrags, nfrags, tmp_dir)
    logger.info('retrieval complete')
    
    logger.info('mering model fragments...')
    merged_embeddings = merge(vecs, order=order, strategy=strategy, uselra=uselra)
    logger.info('merging complete')

    with open(os.path.join(dout, fout_name), 'w+') as fout:
        pickle.dump(dict(zip(common_vocab, merged_embeddings)), fout)
    logger.info("dumping complete")
