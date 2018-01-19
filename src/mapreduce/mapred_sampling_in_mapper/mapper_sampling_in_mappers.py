#!./env/bin/python
from gensim.models.word2vec import Word2Vec

import sys
import copy, random
import logging
import os

nparts = 100
min_count = 100
sub_min_count = min_count/nparts
#sub_min_count=0
n_ns = 5
dim = 500

program = os.path.basename(sys.argv[0])
logger = logging.getLogger(program)                                                                                                                                                                      
logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
logging.root.setLevel(level=logging.INFO)

def reservoir(source, nsamples, mapper_idx):
    ret = []
    for idx, term in enumerate(source):
        if (idx+1)%100 == 0:
            logger.info("mapper #{}, processing sentence #{}".format(mapper_idx, idx))
        if len(ret)<nsamples:
            ret.append(copy.deepcopy(term))
        else:
            if int(random.uniform(0, idx)) < nsamples:
                ret[int(random.uniform(0, nsamples))] = copy.deepcopy(term)
    return [sentence.split() for sentence in ret]

def skip(source, nparts):
    import itertools
    step = nparts
    lines = source.readlines()
    return itertools.islice(lines, start, None, step)


def sampling(source, nparts, mapper_idx):
    source.seek(0)
    nline = sum(1 for line in source.readlines())
    logger.info('mapper #{}: line counting finished'.format(mapper_idx))
    source.seek(0)
    return reservoir(source, nline/nparts, mapper_idx)


class SubLineSentences(object):

    def __init__(self, source, strategy, nparts, mapper_idx):
        self.strategy = strategy
        self.nparts = nparts
        self.source_doc = open(source, 'r')
        self.mapper_idx = mapper_idx
        if strategy=='skip':
            self.sub_set = skip(self.source_doc, self.nparts)
        elif strategy=='sampling':
            self.sub_set = sampling(self.source_doc, self.nparts, self.mapper_idx)

    def __iter__(self):
        for sentence in self.sub_set:
            yield sentence
            
    def __del__(self):
        self.source_doc.close()


if __name__ == '__main__':
    for task in sys.stdin:
        idx, filename = task.split('#')
        idx = idx.strip()
        filename = filename.strip()
        logger.info('mapper #{}, ready for sampling')
        samples = SubLineSentences(filename, 'sampling', nparts, idx)
        '''
        m = Word2Vec(samples,
                    size = dim,
                    negative = n_ns,
                    workers = 18,
                    window = 10,
                    sg = 1,
                    null_word = 1,
                    min_count = sub_min_count,
                    sample = 1e-4)

        for word in m.wv.vocab:
            embedding = repr(word.strip())+'\t' #repr() is needed because of non-ascii chars in corpus, thus the word in output are all like u'word'
            embedding += idx+'#'
            embedding += str(m.wv[word].tolist())[1:-1]
            print embedding
            '''
        print '1\tfinished'
