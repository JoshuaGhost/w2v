# -*- coding: utf-8 -*-
"""
    This script align different manifold and combine them together
    using average along each dimension, both serially and dicotomically
    merge.

    Usage: python align.py <order> <strategy> <folder/contain/sub-models> <dir/combined-model>

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

def find_corr(my, m2):
    Cxy = np.zeros((my.wv.syn0.shape[0], m2.wv.syn0.shape[0]))
    for word in my.wv.vocab.keys():
        if word in m2.wv.vocab:
            Cxy[my.wv.vocab[word].index][m2.wv.vocab[word].index] = 1
    return Cxy


def serial_model_pair(new_model_name, dirpath):
    mns = [dirpath+'article.txt.'+str(i)+'.txt.w2v' for i in range(10)]
    mx = Word2Vec.load(mns[0])
    mx.init_sims(replace=True)
    my = Word2Vec.load(mns[1])
    my.init_sims(replace=True)
    Cxy = find_corr(mx, my)
    mx.wv.syn0norm = yield [mx.wv.syn0norm, my.wv.syn0norm, Cxy]
    for sub_model_name in mns[2:]:
        my = Word2Vec.load(sub_model_name)
        my.init_sims(replace=True)
        Cxy = find_corr(mx, my)
        mx.wv.syn0norm = yield [mx.wv.syn0norm, my.wv.syn0norm, Cxy]
    my.save(new_model_name)


if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)

    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level = logging.INFO)
    logger.info("running %s" % ' '.join(sys.argv))

    if len(sys.argv) < 5:
        print globals()['__doc__'] % locals()
        sys.exit(1)

    order, strategy, sub_models_dir, model_name = sys.argv[1:5]

    ftime = open('result/default_results/time_alignment_combine.txt', 'a+')
    ftime.write('time of alignment and combination with order: %s and strategy: %s\n'% (order, strategy))
    ftime.write('sub-models under %s\n'%sub_models_dir)
    ftime.write('combined models saved as %s\n\n'%model_name)
    
    etime = -ctime()
    if order == 'serial':
        gen_model_pairs = serial_model_pair(model_name, sub_models_dir)
    elif order == 'dicoto':
        gen_model_pairs = dicoto_model_pair(model_name, sub_models_dir)

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
