#!./env/bin/python
from gensim.models.word2vec import Word2Vec
from gensim.models.word2vec import LineSentence

import sys

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
def sample_table(ntotal, nparts):
    num_per_part = ntotal/nparts
    i = 0
    ret = lil_matrix((1,ntotal))
    while i<num_per_part:
        while True:
            p = randint(0, ntotal-1)
            if ret[0, p] == 0:
                ret[0, p] = 1
                break
        i += 1
    return ret


def part_wrt_hit_table(source, hit):
    source.seek(0)
    for i, line in enumerate(source.readlines()):
        if hit[0, i] != 0:
            yield line


def divide_skip(source, nparts):
    import itertools
    step = nparts
    for start in range(nparts):
        lines = source.readlines()
        yield itertools.islice(lines, start, None, step)


def divide_bootstrap(source, nparts):
    source.seek(0)
    nline = sum(1 for line in source)
    for i in range(nparts):
        source.seek(0)
        hit = sample_table(nline, nparts)
        yield part_wrt_hit_table(source, hit)


class DividedLineSentence(object):

    def __init__(self, strategy, npart, source):
        self.strategy = strategy
        self.npart = npart
        self.source_doc = open(source, 'r')
        if strategy=='skip':
            self.all_parts = divide_skip(self.source_doc, self.npart)
            self.sub_folder='skip/'
        elif strategy=='bootstrap':
            self.all_parts = divide_bootstrap(self.source_doc, self.npart)
            self.sub_folder='bootstrap/'

    def __iter__(self):
        self.source_doc.seek(0)
        for part in self.all_parts:
            yield part
            
    def __del__(self):
        self.source_doc.close()
