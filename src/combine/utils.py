import os
from numpy import hstack, vstack
from numpy import array
from numpy import sqrt
from numpy.linalg import eigh
from gensim.models.word2vec import Word2Vec as w2v

from scalable_learning.extrinsic_evaluation.web.embedding import Embedding
from multiprocessing import Pool

new_dim = 500
def read_wv(fname):
    vocab = []
    vecs = []
    for line in open(fname, 'r'):
        word, vec = map(eval, line.split(':'))
        vocab.append(word)
        vecs.append(vec)
    return vocab, vecs

def load_embeddings(folder, filename, extension, norm, arch):
    vocab = []
    vecs = []
    fnames = os.popen('ls '+folder+'/'+filename+"*"+extension+'|grep -v syn1neg|grep -v syn0').read().split()
    if arch == 'mapreduce':
        p = Pool(18)
        wvs = p.map(read_wv, fnames)
        vocab = [wv[0] for wv in wvs]
        vocab = reduce(list.__add__, vocab)
        vecs = [wv[1] for wv in wvs]
        vecs = vstack(array(vecs))
        if norm:
            for i in range(vecs.shape[0]):
                vecs[i, :] /= sqrt((vecs[i, :] ** 2).sum(-1))
        return Embedding.from_dict(dict(zip(vocab, vecs)))

    elif arch == 'local':
        lsresult = os
        ms = [w2v.load(fname) for fname in fnames]
        vocab = reduce(lambda x, y: x.intersection(y), [set(m.wv.vocab.keys()) for m in ms])
        vocab = list(vocab)
        if norm:
            for model in ms:
                model.init_sims()
            vecs = [[m.wv.syn0norm[m.wv.vocab[word].index] for word in vocab] for m in ms]
        else:
            vecs = [[m.wv.syn0[m.wv.vocab[word].index] for word in vocab] for m in ms]
        return Embedding(zip(vocab, hstack(vecs)))


def dim_reduce(vecs, eigval_threshold, mean_corr):
    ms = None
    if mean_corr:
        vecs = vecs-vecs.mean(axis=0)
    cov = vecs.T.dot(vecs)
    evalues, bases = eigh(cov)
    evalues = sqrt(evalues)
    bases = bases[evalues>eigval_threshold]
    vecs = vecs.dot(bases.T)
    return vecs

