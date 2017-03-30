# -*- coding: utf-8 -*-
"""
    This script align different manifold and combine them together
    using average along each dimension, both serially and dicotomically
    merge.

    Usage: python align.py <order> <strategy> <folder/contain/sub-models> <dir/combined-model> <num of submodels>

    For order it can be:
        serial: serial merge
        dicoto: dicotomicall merge

    For strategy it can be:
        avg: merge using average through each dimension
"""

from lra import low_rank_align
from combine import combine_avg

import logging

from gensim.models.word2vec import Word2Vec
from time import time as ctime

import os, sys
from time import time as ctime
import os.path

import numpy as np

from scipy.sparse import lil_matrix

def find_corr(mx, my):
    Cxy = lil_matrix((mx.wv.syn0.shape[0], my.wv.syn0.shape[0]))
    for word in mx.wv.vocab.keys():
        if word in my.wv.vocab:
            Cxy[mx.wv.vocab[word].index,my.wv.vocab[word].index] = 1
    return Cxy


def ebd_inters(n_xsamples, my, vocab_in_x):
    Y = [my.wv.syn0norm[my.wv.vocab[word].index] for word in vocab_in_x if word in my.wv.vocab]
    Y = np.array(Y)
    Cxy = np.zeros((n_xsamples, Y.shape[0]))
    i = 0
    for idx, word in enumerate(vocab_in_x):
        if word in my.wv.vocab:
            Cxy[idx, i] = 1
            i += 1
    return Y, Cxy


def serial_model_pair(mns, vocab, new_model_name):
    mx = Word2Vec.load(mns[0])
    mx.init_sims(replace=True)
    my = Word2Vec.load(mns[1])
    my.init_sims(replace=True)
    vocab_in_x = [word.lower()[:-1] for word in vocab.readlines() if word.lower()[:-1] in mx.wv.vocab]
    idx = []; X = []
    logger.info('first two model loaded successfully')
    logger.info('start to derive embedding matrix X, Y and correspondence matrix Cxy')
    for word in vocab_in_x:
        idx.append(mx.wv.vocab[word].index)
        X.append(mx.wv.syn0norm[idx[-1]])
    X = np.array(X)
    Y, Cxy = ebd_inters(X.shape[0], my, vocab_in_x)
    logger.info('matrices derivation finished, start to align two matrices')
    Z = yield [X, Y, Cxy]
    logger.info('alignment finished')
    for sub_model_name in mns[2:]:
        my = Word2Vec.load(sub_model_name)
        my.init_sims(replace=True)
        Y, Cxy = ebd_inters(X.shape[0], my, vocab_in_x)
        logger.info('new embedding matrxi builded')
        Z = yield [X, Y, Cxy]
        logger.info('alignment finished')
    for i, org_i in enumerate(idx):
        mx.wv.syn0norm[org_i] = Z[i]
    mx.save(new_model_name)



if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])
    global logger
    logger = logging.getLogger(program)

    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level = logging.INFO)
    logger.info("running %s" % ' '.join(sys.argv))

    if len(sys.argv) < 5:
        print globals()['__doc__'] % locals()
        sys.exit(1)

    order, strategy, sub_models_dir, model_name, num_sub_models = sys.argv[1:6]
    
    num_sub_models = int(num_sub_models)

    ftime = open('result/default_results/time_alignment_combine.txt', 'a+')
    ftime.write('time of alignment and combination with order: %s and strategy: %s\n'% (order, strategy))
    ftime.write('sub-models under %s\n'%sub_models_dir)
    ftime.write('combined models saved as %s\n\n'%model_name)
    
    etime = -ctime()

    vocab = open('vocab/vocab.txt', 'r')
    mns = [sub_models_dir+'article.txt.'+str(i)+'.txt.w2v' for i in range(num_sub_models)]
    if order == 'serial':
        gen_model_pairs = serial_model_pair(mns, vocab, model_name)
    elif order == 'dicoto':
        gen_model_pairs = dicoto_model_pair(mns, vocab, model_name)

    i = 0
    Z = None
    while True:
        try:
            X, Y, Cxy = gen_model_pairs.send(Z)
        except StopIteration:
            break
        i+=1
        X, Y = low_rank_align(X, Y, Cxy)
        if strategy == 'avg':
            Z = combine_avg(X, Y, Cxy, i)
    etime += ctime()
    ftime.write('combination time: %f sec\n\n' % etime)
    
    from time import localtime, strftime
    times = strftime("%Y-%m-%d %H:%M:%S", localtime())

    ftime.write("full-stack of experiment finished at %s\n\n" % times)
    ftime.close()
    vocab.close()
