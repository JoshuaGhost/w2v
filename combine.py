# -*- coding: utf-8 -*-
"""
    This script align different manifold and combine them together
    using average along each dimension, both serially and dicotomically
    merge.

    Usage: python combine.py
                <order>
                <with low rank alignment(Y/N)>
                <combination strategy>
                <folder/contain/sub-models>
                <dir/combined-model>
                <num of submodels>

    For order it can be:
        serial: serial merge
        dichoto: dicotomicall merge

    For combination strategy it can be:
        avg: merge using average through each dimension
        avgnorm: merge using average then normalization
        hilbert: merge under hilbert space
"""


import logging
import os
import sys
import os.path
import numpy as np

from gensim import matutils
from gensim.models.word2vec import Word2Vec
from time import time as ctime
from scipy.sparse import lil_matrix
from collections import deque
from glob import glob

from config import LOG_FILE
from lra import low_rank_align


def combine_avg(X, Y, Cxy, times):
    return (X*float(times)+Cxy.dot(Y))/float(times+1)

def combine_avg_norm(X, Y, times):
    Z = combine(X, Y, times)
    Z = [matutils.unitvec(v) for v in Z]
    return Z

def combine_hilbert(X, Y):
    return X + Y

class EmbeddingPairs(object):
    def model_load(self, model_file_name):
        model = Word2Vec.load(model_file_name)
        if self.norm_init:
            model.init_sims()
        return model

    def get_base_vecs(self):
        X = []
        logger.info('start to retrieve embedding matrix')
        X = [self.mx.wv[word] for word in self.vocab]
        X = np.array(X)
        return X

    def get_corresponders(self):
        n_xsamples = self.X.shape[0]
        Y = [self.my.wv[word] for word in self.vocab]
        Y = np.array(Y)
        return Y

    def update_model(self, Z):
        assert Z.shape[0] == len(self.vocab)
        for idx, word in enumerate(self.vocab):
            idx_org = self.mx.wv.vocab[word].index
            if self.norm_init:
                self.mx.wv.syn0norm[idx_org] = Z[idx]
            else:
                self.mx.wv.syn0[idx_org] = Z[idx]

    def serial_pairs_gen(self):
        self.mx = self.model_load(self.mns[0])
        logger.info('base model loaded')
        logger.info('start to combine...')
        self.X = self.get_base_vecs()
        for sub_model_name in mns[1:]:
            self.my = self.model_load(sub_model_name)
            Y = self.get_corresponders()
            logger.info('matrix to merge loaded')
            Z = yield [self.X, Y]
            logger.info('merging finished')
        self.update_model(Z)

    def dichoto_pairs_gen(self):
        limit = len(self.mns)
        combine_queue = deque([str(i) for i in range(limit)])
        idx_model = 0

        def load_using_tag(tag):
            name = (self.temp_models_dir +
                    'article.txt.' +
                    tag + '.txt.w2v')
            model = Word2Vec.load(name)
            if idx_model < self.num_org_models:
                if self.norm_init:
                    model.init_sims(replace=True)
            else:
                for filename in glob(name+'*'):
                    os.remove(filename)
            return model

        while True:
            mx_tag = combine_queue.popleft()
            self.mx = load_using_tag(mx_tag)
            idx_model += 1

            if len(combine_queue) == 0:
                break

            my_tag = combine_queue.popleft()
            self.my = load_using_tag(my_tag)
            idx_model += 1

            self.X = self.get_base_vecs()
            Y = self.get_corresponders()
            Z = yield [self.X, Y]
            self.update_model(Z)

            combine_queue.append(mx_tag+my_tag)
            combined_name = (self.temp_models_dir + 'article.txt.' +
                             mx_tag + my_tag + '.txt.w2v')
            self.mx.save(combined_name, ignore=['cum_table', 'table'])

    def __init__(self, mns, vocab_file, order, temp_models_dir, norm_init):
        self.mns = mns
        self.num_org_models = len(self.mns)
        self.temp_models_dir = temp_models_dir
        with open(vocab_file, 'r') as vocab:
            self.vocab = [word.lower()[:-1] for word in vocab.readlines()]
        for modelname in self.mns:
            mt = Word2Vec.load(modelname)
            self.vocab = [word for word in self.vocab if word in mt.wv.vocab]
        self.X = None
        self.mx = None
        self.my = None
        if order == 'serial':
            self.gen = self.serial_pairs_gen
        if order == 'dichoto':
            self.gen = self.dichoto_pairs_gen
        self.norm_init = norm_init

    def save_final(self, model_dir):
        self.mx.save(model_dir, ignore=['table', 'cum_table'])


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

    (order, use_lra, strategy, sub_models_dir,
     new_model_name, num_sub_models) = sys.argv[1:7]
    num_sub_models = int(num_sub_models)
    if use_lra == 'Y':
        use_lra = True
    else:
        use_lra = False

    log_file = LOG_FILE
    ftime = open(log_file, 'a+')
    ftime.write('time of alignment and combination with order:'
                ' %s and strategy: %s\n' % (order, strategy))
    ftime.write('sub-models under %s\n' % sub_models_dir)
    ftime.write('combined models saved as %s\n\n' % new_model_name)

    etime = -ctime()

    mns = [sub_models_dir+'article.txt.'+str(i)+'.txt.w2v'
           for i in range(num_sub_models)]
    ep = EmbeddingPairs(mns, 'vocab/vocab.txt', order, sub_models_dir, strategy!='hilbert')
    ep_gen = ep.gen()

    i = 0
    Z = None
    while True:
        try:
            X, Y = ep_gen.send(Z)
        except StopIteration:
            break
        i += 1
        if use_lra:
            X, Y = low_rank_align(X, Y, np.eyes(X.shape[0]))
        if strategy == 'avg':
            Z = combine_avg(X, Y, i)
        if strategy == 'avgnorm':
            Z = combine_avgnorm(X, Y, i)
        if strategy == 'hilbert':
            Z = combine_hilbert(X, Y)
    
    ep.save_final(new_model_name)

    etime += ctime()
    ftime.write('combination time: %f sec\n\n' % etime)

    from time import localtime, strftime
    times = strftime("%Y-%m-%d %H:%M:%S", localtime())

    ftime.write("full-stack of experiment finished at %s\n\n" % times)
    ftime.close()

