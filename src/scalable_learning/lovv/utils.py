import codecs
import logging

import numpy as np

from scalable_learning.extrinsic_evaluation.web.embedding import Embedding, OrderedVocabulary


logger = logging.getLogger(__name__)


def gensim2lovv(model, use_norm):
    logger.info('Extracting {}word vectors from gensim model.'.format('nomed ' if use_norm else 'original '))
    vocab = []
    vectors = []
    if use_norm:
        model.init_sims()
    for word in model.wv.vocab.keys():
        assert isinstance(word, str)
        vocab.append(word)
        vectors.append(model.wv.word_vec(word, use_norm=use_norm))
    vvs = sorted(zip(vocab, vectors), key=(lambda x: x[0]))
    vocab = [vv[0] for vv in vvs]
    vectors = np.asarray([vv[1] for vv in vvs])
    return vocab, vectors


def lovv2csv(vv_pair, filename):
    logger.info('dumping word embedding as {}, vocab is sorted by lexical order'.format(filename))
    with codecs.open(filename, 'w+', encoding='utf-8') as fout:
        for word, vector in zip(vv_pair[0], vv_pair[1]):
            fout.write(u'{}, {}\n'.format(word, repr(vector.tolist())[1:-1]))


def lovv2web(vv_pair):
    logger.info('transforming lexical ordered word-vector pairs into web.')
    return Embedding(vocabulary=OrderedVocabulary(vv_pair[0]), vectors=vv_pair[1])
