#!./env/bin/python
from gensim.models.word2vec import Word2Vec
from gensim.models.word2vec import LineSentence

import sys

if __name__ == '__main__':
    #nparts = int(sys.argv[1])
    nparts = 100
    min_count = 100
    sub_min_count = 100/nparts
    n_ns = 5
    dim = 500
    for task in sys.stdin:
        if len(task.strip()) == 0:
            continue
        print task
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
            embedding = repr(word.strip())+'\t' #repr() is needed because of non-ascii chars in corpus, thus the word in output are all like u'word'
            embedding += idx+'#'
            embedding += str(m.wv[word].tolist())[1:-1]
            print embedding
