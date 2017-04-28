from gensim.models.word2vec import Word2Vec
import sys
import numpy as np
import pickle

dump_vectors = False
dim = 500

(frag_folder, basename, num_models, suffix) = sys.argv[1:]
num_models = int(num_models)
#frag_folder = "temp/models/dummy"
#basename = "article.txt."
#suffix = ".txt.w2v"
#num_models = 5

def load_vecs(frag_folder, basename,
              suffix, num_models):
    if dump_vectors:
        with open('vectors.pickle', 'w+') as dfvectors:
            pickle.dump(vec, dfvectors)
    models = [Word2Vec.load(frag_folder +
                            basename +
                            str(i) +
                            suffix)
              for i in range(num_models)]
    vocab_sets = [set(model.wv.vocab) for model in models]
    vocab = reduce(lambda x,y: x.intersection(y), vocab_sets)
    X = [[model.wv[v] for v in vocab] for model in models]
    X = [np.array(vecs) for vecs in X]
    X = reduce(lambda x,y: np.concatenate((x,y), axis=1), X)
    return X

def encode(vec, dim, batch=50, num_iter=5, learning_rate=0.00002):
    num_vocab, dim_total = vec.shape
    num_models = dim_total/dim
    
    we = np.random.normal(size=(dim_total, dim))
    be = np.zeros((1, dim))
    wd = np.random.normal(size=(dim, dim_total))
    bd = np.zeros((1, dim_total))
    for block in range(num_vocab / batch):
        X = vec[block*batch:block*batch+batch, :]
        for iter_round in range(num_iter):
            Z = (X.dot(we) + np.vstack(be for i in range(batch)))#batch*dim
            Y = (Z.dot(wd) + np.vstack(bd for i in range(batch)))#batch*dim_total
            err = (Y - X) / batch#batch*dim_total
            bd -= learning_rate * err.sum(axis=0)
            wd -= learning_rate * (Z.T.dot(err))
            be -= learning_rate * err.sum(axis=0).dot(wd.T)
            we -= learning_rate * X.T.dot(err).dot(wd.T)
            print('batch = %d, num_time = %d, block = %d, iter_round = %d, err = %f' %
                  (batch, num_iter, block, iter_round, (err**2).sum().sum()))
    return we, be, (err**2).sum().sum()


if __name__=='__main__':
    vec = load_vecs(frag_folder, basename,
                    suffix, num_models)

    bi = [(b, i) for b in range(10, 110, 10) for i in range(1, 23, 2)]
    bi.extend([(b, i) for b in range(200, 1100, 100) for i in range(1, 23, 2)])
    err_min = 999999999.0

    for batch, num_iter in bi:
        we, be, err = encode(vec, dim, batch, num_iter)
        if err < err_min:
            (batch_opt, iter_opt, err_min) =\
            (batch, num_iter, err)

    print ("optimized batch: %d\n\
            optimized iter: %d\n\
            minimum error: %f" %
            (batch_opt, iter_opt, err_min))

    #output = vec.dot(we) + np.vstack(be for i in range(num_vocab))

    #with open('vocab.pickle', 'w+') as dfvocab:
    #   pickle.dump(vocab, dfvocab)
    #with open('output.pickle', 'w+') as dfoutput:
    #   pickle.dump(output, dfoutput)


