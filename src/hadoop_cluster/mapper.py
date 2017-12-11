#!./env/bin/python
from gensim.models.word2vec import Word2Vec
from gensim.models.word2vec import LineSentence

import sys

nparts = 100
min_count = 100
sub_min_count = min_count/nparts
#sub_min_count=0
n_ns = 5
dim = 5000/nparts

if __name__ == '__main__':
    for task in sys.stdin:
        idx, filename = task.split('#')
        idx = idx.strip()
        filename = filename.strip()
        m = Word2Vec(LineSentence(filename),
                    size = dim,
                    negative = n_ns,
                    workers = 18,
                    window = 10,
                    sg = 1,
                    null_word = 1,
                    min_count = sub_min_count,
                    sample = 1e-4)

        for word in m.wv.vocab:
            embedding = word.strip()+'\t'
            embedding += idx+'#'
            embedding += str(m.wv[word].tolist())[1:-1]
            print embedding
