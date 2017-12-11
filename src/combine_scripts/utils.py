from numpy import hstack
from numpy import array
from numpy.linalg import eigh
from gensim.models.word2vec import Word2Vec as w2v

nmodels = 10
new_dim = 500

def main(folder_in, filename, extension, norm, mean_corr, folder_out, dump_name):
    vocab, vecs = load_embeddings(folder, filename, extension, nmodels)
    vocab, vecs = dim_reduce(vocab, vecs, new_dim, mean_corr) 
    d = dict(zip(vocab, vecs))
    dump(d, open(dump_name+'.pkl', 'w+'))

def load_embeddings(folder, filename, extension, nmodels, norm):
    ms=[w2v.load(folder+'/'+filename+str(i)+extension) for i in range(nmodels)]
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
    return dict(zip(vocab, vecs))

