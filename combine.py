# -*- coding: utf-8 -*-
"""
    This script align different manifold and combine them together
    using average along each dimension, both serially and dicotomically
    merge.

    Usage: python combine.py <order> <strategy> <folder/contain/sub-models> <dir/combined-model> <num of submodels>

    For order it can be:
        serial: serial merge
        dichoto: dicotomicall merge

    For strategy it can be:
        avg: merge using average through each dimension
"""


import logging
import os, sys
import os.path
import numpy as np

from gensim.models.word2vec import Word2Vec
from time import time as ctime
from scipy.sparse import lil_matrix
from collections import deque
from glob import glob

from config import LOG_FILE
from lra import low_rank_align


def combine_avg(X, Y, Cxy, times):
    return (X*float(times)+Cxy.dot(Y))/float(times+1)


def find_corr(mx, my):
    Cxy = lil_matrix((mx.wv.syn0.shape[0], my.wv.syn0.shape[0]))
    for word in mx.wv.vocab.keys():
        if word in my.wv.vocab:
            Cxy[mx.wv.vocab[word].index,my.wv.vocab[word].index] = 1
    return Cxy


def intersect_embedding_in_Y(X, my, vocab):
    n_xsamples = X.shape[0]
    vocab_idx_in_y = [my.wv.vocab[word].index for word in vocab]
    Y = [my.wv.syn0norm[i] for i in vocab_idx_in_y]
    Y = np.array(Y)
    Cxy = np.zeros((X.shape[0], Y.shape[0]))
    i = 0
    for idx, word in enumerate(vocab):
        if word in my.wv.vocab:
            Cxy[idx, i] = 1
            i += 1
    return Y, Cxy


def extract_vectors_according_vocab(model, vocab):
    idx = []; X = []
    logger.info('first two model loaded successfully')
    logger.info('start to derive embedding matrix X, Y and correspondence matrix Cxy')
    for word in vocab:
        idx.append(model.wv.vocab[word].index)
        X.append(model.wv.syn0norm[idx[-1]])
    X = np.array(X)
    return X, idx


def load_using_tag(tag, temp_models_dir, idx_model, limit):
    name = temp_models_dir+'article.txt.'+tag+'.txt.w2v'
    model = Word2Vec.load(name)
    if idx_model < limit:
        model.init_sims(replace = True)
    else:
        for filename in glob(name+'*'):
            os.remove(filename)
    return model



class Pairs_generator():
    def __init__(self, mns, vocab, order, temp_models_dir):
        self.mns = mns
        self.vocab = [word.lower()[:-1] for word in vocab.readlines()]
        for modelname in self.mns:
            mt = Word2Vec.load(modelname)
            self.vocab = [word for word in self.vocab if word in mt.wv.vocab]

        if order == 'serial':
            self.gen = self.serial_pairs_gen()
        if order == 'dichoto':
            self.gen = self.dichoto_pairs_gen(temp_models_dir)


    def dichoto_pairs_gen(self, temp_models_dir):
        limit = len(self.mns)
        combine_queue = deque([str(i) for i in range(limit)])
        idx_model = 0
        while True:
            mx_tag = combine_queue.popleft()
            mx = load_using_tag(mx_tag,\
                                temp_models_dir,\
                                idx_model,\
                                limit)
            idx_model+=1

            if len(combine_queue) == 0:
                self.mx = mx
                break

            my_tag = combine_queue.popleft()
            my = load_using_tag(my_tag,\
                                temp_models_dir,\
                                idx_model,\
                                limit)
            idx_model+=1
            
            X, idx = extract_vectors_according_vocab(mx, self.vocab)
            Y, Cxy = intersect_embedding_in_Y(X, my, self.vocab)
            Z = yield [X, Y, Cxy]
            for i, org_i in enumerate(idx):
                mx.wv.syn0norm[org_i] = Z[i]

            combine_queue.append(mx_tag+my_tag)
            combined_name = temp_models_dir+'article.txt.'+mx_tag+my_tag+'.txt.w2v'
            mx.save(combined_name, ignore = ['cum_table', 'table'])


    def serial_pairs_gen(self):
        mx = Word2Vec.load(self.mns[0])
        my = Word2Vec.load(self.mns[1])
        mx.init_sims(replace = True)
        my.init_sims(replace = True)

        logger.info('first two model loaded successfully')
        logger.info('start to derive embedding matrix X, Y and correspondence matrix Cxy')
        X, idx = extract_vectors_according_vocab(mx, self.vocab)
        logger.info('matrices derivation finished, start to align two matrices')
        Y, Cxy = intersect_embedding_in_Y(X, my, self.vocab)
        Z = yield [X, Y, Cxy]
        logger.info('alignment finished')
        for sub_model_name in mns[2:]:
            my = Word2Vec.load(sub_model_name)
            my.init_sims(replace=True)
            Y, Cxy = intersect_embedding_in_Y(X, my, self.vocab)
            logger.info('new embedding matrx builded')
            Z = yield [X, Y, Cxy]
            logger.info('alignment finished')
        for i, org_i in enumerate(idx):
            mx.wv.syn0norm[org_i] = Z[i]
        self.mx = mx


    def save_final(self, model_dir):
        self.mx.save(model_dir, ignore = ['table', 'cum_table'])



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

    order, strategy, sub_models_dir, new_model_name, num_sub_models = sys.argv[1:6]
    num_sub_models = int(num_sub_models)
    
    log_file = LOG_FILE
    ftime = open(log_file, 'a+')
    ftime.write('time of alignment and combination with order: %s and strategy: %s\n'% (order, strategy))
    ftime.write('sub-models under %s\n'%sub_models_dir)
    ftime.write('combined models saved as %s\n\n'%new_model_name)
    
    etime = -ctime()

    vocab = open('vocab/vocab.txt', 'r')
    mns = [sub_models_dir+'article.txt.'+str(i)+'.txt.w2v' for i in range(num_sub_models)]
    vp = Pairs_generator(mns, vocab, order, sub_models_dir)
    vp_gen = vp.gen

    i = 0; Z = None
    while True:
        try:
            X, Y, Cxy = vp_gen.send(Z)
        except StopIteration:
            break
        i+=1
        X, Y = low_rank_align(X, Y, Cxy)
        if strategy == 'avg':
            Z = combine_avg(X, Y, Cxy, i)
    vp.save_final(new_model_name)

    etime += ctime()
    ftime.write('combination time: %f sec\n\n' % etime)
    
    from time import localtime, strftime
    times = strftime("%Y-%m-%d %H:%M:%S", localtime())

    ftime.write("full-stack of experiment finished at %s\n\n" % times)
    ftime.close()
    vocab.close()
