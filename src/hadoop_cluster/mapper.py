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
             self.sentences = self.divide_skip(source, npart)
             self.tempdir='skip/'
         elif strategy=='sampeling':
             self.sentences = self.divide_bootstrap(source, npart)
             self.tempdir='sampeling/'
 
     def __iter__(self):
         self.source.seek(0)
         for sentence in self.sentences:
             yield sentence
 
     def __del__(self):
         self.source_doc.close()      
#==========================================not completed above========================
for line in sys.stdin:
    order, filename = line.split('#')
    order = order.strip()
    filename = filename.strip()
    m = Word2Vec(LineSentence(filename.strip()), min_count=0)
    for word in m.wv.vocab:
        print word.strip() + '\t' + order + '#' +\
              ','.join([str(num) for num in m.wv[word]]).strip()
