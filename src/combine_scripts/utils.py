import os
from numpy import hstack
from numpy import array
from numpy.linalg import eigh
from gensim.models.word2vec import Word2Vec as w2v

from multiprocessing import Pool

nmodels = 10
new_dim = 500
def read_wv(fname):
    for line in open(fname, 'r'):
        word, vec = map(eval, line.split(':'))
    return word, vec

def load_embeddings(folder, filename, extension, nmodels, norm, arch):
    vocab = []
    vecs = []
    fnames = os.popen('ls '+folder+filename+"*"+extension).read().split()
    if arch == 'mapreduce':
        p = Pool(18)
        wvs = p.map(read_wv, fnames)
        vocab = [wv[0] for wv in wvs]
        vecs = [wv[1] for wv in wvs]
        print len(vocab)
        print len(vecs)
        print vocab[0]
        print vecs[0][0]
        return vocab, array(vecs)

    elif arch == 'local':
        lsresult = os
        ms=[w2v.load(fname) for fname in fnames]
        vocab = reduce(lambda x, y: x.intersection(y), [set(m.wv.vocab.keys()) for m in ms])
        vocab = list(vocab)
        if norm:
            for model in ms:
                model.init_sims()
            vecs = [[m.wv.syn0norm[m.wv.vocab[word].index] for word in vocab] for m in ms]
        else:
            vecs = [[m.wv.syn0[m.wv.vocab[word].index] for word in vocab] for m in ms]
        return vocab, hstack(vecs)

def dim_reduce(vocab, vecs, new_dim, mean_corr):
    ms = None
    if mean_corr:
        vecs_mean_corr = vecs-(vecs.sum(axis=0).reshape((1,vecs.shape[1])))/vecs.shape[0]
    cov = vecs.transpose().dot(vecs)
    evalues, bases = eigh(cov)
    bases = bases[-new_dim:]
    evalues = evalues[-new_dim:]
    vecs = array(vecs)
    vecs = vecs.dot(bases.transpose())
    return vocab, vecs

