from gensim.models.word2vec import Word2Vec
import sys
import numpy as np
import pickle

dump_vectors = False
non_linear = True
dim = 500

batch = 1000
num_iter = 21

(frag_folder, basename, num_models, suffix, non_linear, output_folder) = sys.argv[1:]
num_models = int(num_models)
non_linear = True if non_linear == "True" else False

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
    vocab = list(vocab)
    X = [[model.wv[v] for v in vocab] for model in models]
    X = [np.array(vecs) for vecs in X]
    X = reduce(lambda x,y: np.concatenate((x,y), axis=1), X)
    return vocab, X

def encode(vec, dim, batch=50, num_iter=500, learning_rate=0.00001):
    num_vocab, dim_total = vec.shape
    num_models = dim_total/dim

    we = np.random.normal(size=(dim_total, dim))
    be = np.zeros((1, dim))
    wd = np.random.normal(size=(dim, dim_total))
    bd = np.zeros((1, dim_total))

    def linear_train(X, we, be, wd, bd):
        Z = (X.dot(we) + np.vstack(be for i in range(batch)))#batch*dim
        Y = (Z.dot(wd) + np.vstack(bd for i in range(batch)))#batch*dim_total
        err = (Y - X) / batch#batch*dim_total

        wd -= learning_rate * (Z.T.dot(err))
        bd -= learning_rate * err.sum(axis=0)
        we -= learning_rate * X.T.dot(err).dot(wd.T)
        be -= learning_rate * err.dot(wd.T).sum(axis=0)

        return we, be, wd, bd, err

    def non_linear_train(X, we, be, wd, bd):
        Z = (X.dot(we) + np.vstack(be for i in range(batch)))#batch*dim
        Ztanh = np.tanh(Z)
        Y = (Ztanh.dot(wd) + np.vstack(bd for i in range(batch)))#batch*dim_total
        Ytanh = np.tanh(Y)
        err = (Ytanh - X) / batch#batch*dim_total

        errty = err * 2 * Ytanh * (1 - Ytanh)#batch*dim_total
        errtywdtz = errty.dot(wd.T) * 2 * Ztanh * (1 - Ztanh)#batch*dim
        bd -= learning_rate * errty.sum(axis=0)
        wd -= learning_rate * Z.T.dot(errty)#dim*dim_total
        be -= learning_rate * errtywdtz.sum(axis=0)
        we -= learning_rate * X.T.dot(errtywdtz)

        return we, be, wd, bd, err

    for block in range(num_vocab / batch):
        X = vec[block*batch:block*batch+batch, :]
        for iter_round in range(num_iter):
            if non_linear:
                we, be, wd, bd, err = non_linear_train(X, we, be, wd, bd)
            else:
                we, be, wd, bd, err = linear_train(X, we, be, wd, bd)

            print('batch = %d, num_time = %d, block = %d, iter_round = %d, err = %f' %
                  (batch, num_iter, block, iter_round, (err**2).sum().sum()))
    return we, be, (err**2).sum().sum()


if __name__=='__main__':
    vocab, vec = load_vecs(frag_folder, basename,
                    suffix, num_models)

    we, be, err = encode(vec, dim, batch, num_iter)
    output = vec.dot(we) + np.vstack(be for i in range(vec.shape[0]))
    if non_linear:
        output = np.tanh(output)

    with open(output_folder + '/vocab.pickle', 'w+') as dfvocab:
       pickle.dump(vocab, dfvocab)
    with open(output_folder + 'output.pickle', 'w+') as dfoutput:
       pickle.dump(output, dfoutput)

