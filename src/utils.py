import codecs
import logging
import os

import numpy as np
from gensim.models.word2vec import LineSentence, Word2Vec
from gensim.utils import RULE_KEEP, RULE_DISCARD

from config import *
from scalable_learning.extrinsic_evaluation.web.embedding import Embedding, OrderedVocabulary
from scalable_learning.lovv.utils import gensim2lovv, lovv2csv
from scalable_learning.utils import load_embeddings

logger = logging.getLogger(__name__)

def load_or_train(fname_corpus, size=DIM, filter_vocab=None, negative=N_NS, workers=NUM_WORKERS,
                  min_count=SUB_MIN_COUNT, use_norm=True, dump_format='csv', dump_file='./dump.csv',
                  output_format='lovv'):
    if os.path.isfile(dump_file):
        logger.info('loading dumped model: {}'.format(dump_file))
        model = load_embeddings(folder='/', filename=dump_file, extension='', use_norm=True, input_format=dump_format,
                                output_format=output_format)[0]
    else:
        logger.info('training new model')
        if filter_vocab is not None:
            trim_rule = lambda word, count, min_count: RULE_DISCARD if word in filter_vocab or count < min_count \
                                                                            else RULE_KEEP
        else:
            trim_rule = None
        model = Word2Vec(size=size, negative=negative, workers=workers, window=10, sg=1, null_word=1,
                         min_count=min_count, sample=1e-4)
        corpus = LineSentence(fname_corpus)
        model.build_vocab(corpus, trim_rule=trim_rule)
        model.train(corpus, total_examples=model.corpus_count, epochs=model.epochs)
        model.init_sims()
        if output_format == 'lovv':
            model = gensim2lovv(model, use_norm=use_norm)
            lovv2csv(model, dump_file)
        elif output_format == 'web':
            model = gensim2web(model, use_norm=use_norm)
            web2csv(model, dump_file)
        logger.info('source model dumped as {}'.format(dump_file))
    return model


def common_words_index(part, full):
    pp = 0
    pf = 0
    while pp<len(part) and pf<len(full):
        if part[pp]<full[pf]:
            pp += 1
        elif part[pp]>full[pf]:
            pf += 1
        else:
            yield(pf)
            pp += 1
            pf += 1


def gensim2web(model):
    vocabulary = []
    vectors = []
    for word in model.wv.vocab.keys():
        vocabulary.append(word)
        vectors.append(model.wv.word_vec(word, use_norm=True))
    vocabulary = OrderedVocabulary(vocabulary)
    vectors = np.asarray(vectors)
    return Embedding(vocabulary=vocabulary, vectors=vectors)


def web2csv(web, filename):
    print("dumping word embedding under {}".format(filename))
    with codecs.open(filename, 'w+', encoding='utf-8') as fout:
        for word in web.vocabulary:
            fout.write(u'{}, {}\n'.format(word, repr(web[word].tolist())[1:-1]))

