import os
from numpy import hstack, vstack
from numpy import sqrt
from gensim.models.word2vec import Word2Vec
from functools import reduce

from multiprocessing import Pool

import codecs
import numpy as np

from scalable_learning.extrinsic_evaluation.web.embedding import Embedding
from scalable_learning.extrinsic_evaluation.web.vocabulary import OrderedVocabulary

new_dim = 500


def read_wv_hadoop(fname):
    vocab, vecs = [], []
    for line in open(fname, 'r'):
        word, vec = map(eval, line.split(':'))
        vocab.append(word)
        vecs.append(vec)
    vecs = np.array(vecs, dtype=float)
    for i in range(vecs.shape[0]):
        vecs[i, :] /= sqrt((vecs[i, :]**2).sum(-1))
    vocab = OrderedVocabulary(words=vocab)
    return Embedding(vocabulary=vocab, vectors=vecs)


def read_wv_csv(fname):
    vocab, vecs = [], []
    vocab_set = set()
    for idx, line in enumerate(codecs.open(fname, 'r', buffering=(2<<16+8), encoding='utf-8')):
        if len(line.strip()) == 0:
            break
        word, vec = line.split(',', 1)

        if word[0] == '0':
            word = word[4:]# this is set when the first field is index of mapper

        vec = eval(vec)
        vocab.append(word)
        vecs.append(vec)
        if idx % 10000 == 0:
            print("{}: len(word) = {}, len(vecs) = {}".format(idx, len(vocab), len(vecs)))
    vecs = np.asarray(vecs, dtype=float)
    print("len(vocab) = {}, len(vecs) = {}".format(len(vocab), len(vecs)))
    for i in range(vecs.shape[0]):
        vecs[i, :] /= sqrt((vecs[i, :]**2).sum(-1)) # normalization
    vocab = OrderedVocabulary(words=vocab)
    return Embedding(vocabulary=vocab, vectors=vecs)


def load_embeddings(folder, filename, extension, norm, arch, selected=None):
    fnames = os.popen('ls '+folder+'/'+filename+"*"+extension+'|grep -v syn1neg|grep -v syn0').read().split()
    dim_sub = 50
    if arch == 'hadoop':
        p = Pool(18)
        ebs = p.map(read_wv_hadoop, fnames)
        vocab = [embedding.vocabulary for embedding in ebs]
        vecs = [embedding.vectors for embedding in ebs]
        vocab = reduce(lambda x, y: x + y, vocab)
        vecs = vstack(vecs)
        wvs = [Embedding(vocabulary=OrderedVocabulary(vocab), vectors=vecs[:, offset:offset+dim_sub])
               for offset in range(0, vecs.shape[-1], dim_sub)]
        return wvs
    elif arch == 'local':
        ms = [Word2Vec.load(fname) for fname in fnames]
        wvs = []
        for m in ms:
            vocab = list(m.vocab.keys())
            if norm:
                m.init_sims()
                vecs = np.array(m.wv.syn0norm[m.wv.vocab[word].index] for word in vocab)
            else:
                vecs = np.array(m.wv.syn0[m.wv.vocab[word].index] for word in vocab)
            vocab = OrderedVocabulary(vocab)
            wvs.append(Embedding(vocabulary=vocab, vectors=vecs))
        return wvs
    elif arch == 'submodels':
        p = Pool(10)
        fnames = [folder+'/'+filename+str(i)+extension for i in selected]
        wvs = p.map(read_wv_hadoop, fnames)
        return wvs
    elif arch == 'csv':
        p = Pool(10)
        wvs = p.map(read_wv_csv, fnames)
        return wvs

def gensim2web(model):
    vocabulary = []
    vectors = []
    for word in model.wv.vocab.keys():
        vocabulary.append(word)
        vectors.append(model.wv.word_vec(word, use_norm=True))
    vocabulary = OrderedVocabulary(vocabulary)
    vectors = np.asarray(vectors)
    return Embedding(vocabulary=vocabulary, vectors=vectors)