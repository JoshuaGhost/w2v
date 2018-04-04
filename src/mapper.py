#!./env/bin/python
from gensim.models.word2vec import Word2Vec
from gensim.models.word2vec import LineSentence

import codecs
import sys

min_count = 100
n_ns = 5

UTF8Reader = codecs.getreader('utf-8')
sys.stdin = UTF8Reader(sys.stdin)
UTF8Writer = codecs.getwriter('utf-8')
sys.stdout = UTF8Writer(sys.stdout)

if __name__ == '__main__':
    nparts, dim = map(int, sys.argv[1:])
    sub_min_count = min_count/nparts
    for line in sys.stdin:
        mapper_id, sub_corpus = line.split(':')
        sub_corpus = sub_corpus.split('?')
        sub_corpus = [[word.strip() for word in sentence.strip().split()] for sentence in sub_corpus]
        m = Word2Vec(sub_corpus, size = dim, negative = n_ns,
                    workers = 16, window = 10, sg = 1, null_word = 1,  # notice that worker was original 18, now 16
                    min_count = sub_min_count, sample = 1e-4)

        for word in m.wv.vocab:
            #wv = repr(word.strip()) #repr() is needed because of non-ascii chars in corpus, thus the word in output are all like u'word'
            wv = word + ', '+str(m.wv[word].tolist())[1:-1]  # from 1 to -1 because the string representation of array contains pair of squared brackets
            print mapper_id + '\t' + wv
