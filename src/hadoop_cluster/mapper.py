#!./env/bin/python
from gensim.models.word2vec import Word2Vec
from gensim.models.word2vec import LineSentence

import sys

for line in sys.stdin:
    m = Word2Vec(LineSentence(line.strip()), min_count=0)
    for word in m.wv.vocab:
        print '\t'.join((word, ','.join([str(num) for num in m.wv[word]])))
