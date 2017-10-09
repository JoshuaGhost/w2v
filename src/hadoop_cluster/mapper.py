#!./env/bin/python
from gensim.models.word2vec import Word2Vec
from gensim.models.word2vec import LineSentence

import sys

class DividedLineSentence(object):
    def __init__(self, source, strategy='sampling', npart=10):
        self.strategy = strategy
        self.npart = npart
        self.source = open(source, 'r')
        if strategy=='skip':
            self.centences = self.divide_skip(source, npart)
            self.tempdir='skip/'
        elif strategy=='sampeling':
            self.centences = self.divide_bootstrap(source, npart)
            self.tempdir='sampeling/'

    def __iter__(self):
        self.source.seek(0)
        for centence in self.centences:
            yield centence

    def __del__(self):
        self.source_doc.close()

if __name__ == '__main__':
    for line in sys.stdin:
        m = Word2Vec(DividedLieSentence(line.strip()),
                        size = 500,
                        negative = 5,
                        workers = 18,
                        window = 10,
                        sg = 1,
                        null_word = 1,
                        min_count = new_min_count,
                        sample = 1e-4)

    for word in m.wv.vocab:
        print '\t'.join((word, ','.join([str(num) for num in m.wv[word]])))
