#!./env/bin/python
from gensim.models.word2vec import Word2Vec
from gensim.models.word2vec import LineSentence

import sys
import copy, random

nparts = 100
min_count = 100
sub_min_count = min_count/nparts
#sub_min_count=0
n_ns = 5
dim = 500


def reservoir(source, nsamples):
    ret = []
    for idx, term in enumerate(source):
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


def sampling(source, nparts):
    source.seek(0)
    nline = sum(1 for line in source.readlines())
    source.seek(0)
    return reservoir(source, nline/nparts)


class SubLineSentences(object):

    def __init__(self, source, strategy, nparts):
        self.strategy = strategy
        self.nparts = nparts
        self.source_doc = open(source, 'r')
        if strategy=='skip':
            self.sub_set = skip(self.source_doc, self.nparts)
        elif strategy=='sampling':
            self.sub_set = sampling(self.source_doc, self.nparts)

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
        samples = SubLineSentences(filename, 'sampling', nparts)
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
            print word.strip() + '\t' + idx + '#' +\
                  ','.join([str(num) for num in m.wv[word]]).strip()
