import codecs
import logging
import os
from functools import reduce
from multiprocessing import Pool

import numpy as np
from gensim.models.word2vec import Word2Vec
from numpy import vstack
from numpy.linalg import norm

from scalable_learning.extrinsic_evaluation.web.embedding import Embedding
from scalable_learning.extrinsic_evaluation.web.vocabulary import OrderedVocabulary
from scalable_learning.lovv.utils import gensim2lovv


logger = logging.getLogger()


def read_wv_hadoop(fname):
    vocab, vecs = [], []
    for line in open(fname, 'r'):
        word, vec = map(eval, line.split(':'))
        vocab.append(word)
        vecs.append(vec)
    vecs = np.asarray(vecs, dtype=float)
    return vocab, vecs


def read_wv_csv(fname):
    vocab, vecs = [], []
    vocab_set = set()
    fin = codecs.open(fname, 'r', buffering=(2 << 16 + 8), encoding='utf-8')
    for idx, line in enumerate(fin):
        if len(line.strip()) == 0:
            break
        word, vec = line.split(',', 1)

        if word[0] == '0':
            word = word[4:]  # this is set when the first field is index of mapper

        vec = eval(vec)
        vocab.append(word)
        vecs.append(vec)
        if idx % 10000 == 0:
            logger.info("{}: len(word) = {}, len(vecs) = {}".format(idx, len(vocab), len(vecs)))
    fin.close()
    vecs = np.asarray(vecs, dtype=float)
    logger.info("loading finished, {} word vectors loaded".format(len(vocab)))
    return vocab, vecs


def normalize(vector):
    return vector/norm(vector)


def load_embeddings(folder, filename, extension, use_norm, input_format,
                    vocab_wanted=None, output_format=None, dim_sub=50):
    fnames = os.popen(
        'ls ' + folder + '/' + filename + "*" + extension + '|grep -v syn1neg|grep -v syn0').read().split()
    if input_format == 'hadoop':
        p = Pool(18)
        vvpairs = p.map(read_wv_hadoop, fnames)
        vocabs = [vv[0] for vv in vvpairs]
        vecses = [vv[1] for vv in vvpairs]
        vocab = reduce(lambda x, y: x + y, vocabs)
        vecs = vstack(vecses)
        vvs = [(vocab, vecs[:, offset:offset + dim_sub]) for offset in range(0, vecs.shape[-1], dim_sub)]
    elif input_format == 'local':
        ms = [Word2Vec.load(fname) for fname in fnames]
        vvs = []
        for m in ms:
            vocab = list(m.vocab.keys())
            if use_norm:
                m.init_sims()
                vecs = np.array(m.wv.syn0norm[m.wv.vocab[word].index] for word in vocab)
            else:
                vecs = np.array(m.wv.syn0[m.wv.vocab[word].index] for word in vocab)
            vvs.append((vocab, vecs))
    elif input_format == 'submodels':
        p = Pool(18)
        fnames = [folder + '/' + filename + str(i) + extension for i in vocab_wanted]
        vvs = list(p.map(read_wv_hadoop, fnames))
        if use_norm:
            vvs = [(vocab, np.asarray(p.map(normalize, vecs))) for vocab, vecs in vvs]
    elif input_format == 'csv':
        logger.info('loading embeddings from csv files: {}'.format(fnames))
        p = Pool(18)
        vvs = list(p.map(read_wv_csv, fnames))
        if use_norm:
            logger.info('normalizing the word-vectors')
            vvs = [(vocab, np.asarray(p.map(normalize, vecs))) for vocab, vecs in vvs]
    if output_format is None:
        return [Embedding(vocabulary=OrderedVocabulary(vocab), vectors=vecs) for vocab, vecs in vvs]
    elif output_format == 'lovv':
        vvs = [sorted(zip(vocab, vecs), key=(lambda x:x[0])) for vocab, vecs in vvs]
        vocabs = [[wv[0] for wv in vv] for vv in vvs]
        vectorses = [np.asarray(list(wv[1] for wv in vv)) for vv in vvs]
        return list(zip(vocabs, vectorses))


def gensim2web(model, use_norm):
    vocab, vectors = gensim2lovv(model, use_norm)
    vocabulary = OrderedVocabulary(vocab)
    return Embedding(vocabulary=vocabulary, vectors=vectors)


def web2csv(web, filename):
    logger.info("dumping word embedding as {}".format(filename))
    with codecs.open(filename, 'w+', encoding='utf-8') as fout:
        for word in web.vocabulary:
            fout.write(u'{}, {}\n'.format(word, repr(web[word].tolist())[1:-1]))
