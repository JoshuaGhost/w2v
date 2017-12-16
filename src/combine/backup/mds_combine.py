from gensim.models.word2vec import Word2Vec as w2v

filename = 'article.txt'
extension = 'txt.w2v'


from multiprocessing import Pool
pool = Pool(10)
#ms = pool.map(lambda model_name: w2v.load(model_name), [filename+'.'+str(i)+'.'+extension for i in range(10)])
ms = [w2v.load(filename+'.'+str(i)+'.'+extension) for i in xrange(10)]

vocab = reduce(lambda x, y: x.intersection(y), [set(m.wv.vocab.keys()) for m in ms])
vocab = list(vocab)

#vecs = pool.map(lambda m: [m.wv[word] for word in vocab], ms)
#pool.close()
vecs = [[m.wv[word] for word in vocab] for m in ms]

ms = None

from numpy import hstack
vecs = hstack(vecs)
corr = vecs.transpose().dot(vecs)

from numpy.linalg import eigh
evalues, bases = eigh(corr)
bases = bases[-500:]
evalues = evalues[-500:]
from numpy import array
vecs = array(vecs)
vecs = vecs.dot(bases.transpose())

'''
evalues.reshape((500, 1))
from numpy import ones, eyes, abs
evalues = evalues.dot(ones((1,500)))
evalues.*eyes(500)
print abs(evalues.dot(base)-vecs).sum()
print abs(evalues.)
'''

d = dict(zip(vocab, vecs))

from pickle import dump
dump(d, open('mds.pkl', 'w+'))

