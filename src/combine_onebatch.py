from gensim.models.word2vec import Word2Vec
import sys
import numpy as np
import pickle

dump_vectors = False
non_linear = True
dim = 500

batch = 1000
num_iter = 500


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

def encode(vec, dim, linear, with_bias, batch=50, num_iter=500, learning_rate=0.00001):
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

    def linear_train_no_offset(X, we, be, wd, bd):
        Z = X.dot(we)
        Y = Z.dot(wd)

        err = (Y-X)/batch
        wd -= learning_rate * Z.T.dot(err)
        we -= learning_rate * X.T.dot(err).dot(wd.T)
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
            if not linear:
                we, be, wd, bd, err = non_linear_train(X, we, be, wd, bd)
            elif with_bias:
                we, be, wd, bd, err = linear_train(X, we, be, wd, bd)
            else:
                we, be, wd, bd, err = linear_train_no_offset(X, we, be, wd, bd)

            print('batch = %d, num_iter = %d, block = %d, iter_round = %d, err = %f' %
                  (batch, num_iter, block, iter_round, (err**2).sum().sum()))
    return we, be, (err**2).sum().sum()

def pca(vec, dim, bias_method):
    if bias_method == 'with':
        R = vec.T.dot(vec)
    else:
        bias = vec.mean(axis=0)
        R = np.cov(vec, rowvar=False)
    
    assert R.shape == (vec.shape[1], vec.shape[1])
    _, evecs = np.linalg.eigh(R)
    evecs = evecs[:, -dim:]
    output = vec.dot(evecs)

    if bias_method == 'separate':
        bias = bias.dot(evecs)
        output += np.ones((vec.shape[0], 1)).dot(bias.reshape((1, dim)))
    return output
    

if __name__=='__main__':
    (frag_folder, basename, num_models, suffix, method, output_folder) = sys.argv[1:]
    num_models = int(num_models)

    method = method.split('-')
    vocab, vec = load_vecs(frag_folder, basename,
                           suffix, num_models)
 
    if method[0] == 'autoencoder':
        if method[1] == 'linear':
            linear = True
        elif method[1] == 'non_linear':
            linear = False
        if method[2] == 'with_bias':
            with_bias = True
        elif method[2] == 'without_bias':
            with_bias = False
        we, be, err = encode(vec, dim, linear, with_bias, batch, num_iter)
        output = vec.dot(we) + np.vstack(be for i in range(vec.shape[0]))
        if not linear:
            output = np.tanh(output)

    elif method[0] == 'pca':
        bias_method = method[1].split('_')[0]#with_bias, ignore_bias, separate_bias
        output = pca(vec, dim, bias_method)

    with open(output_folder + '/vocab.pickle', 'w+') as dfvocab:
       pickle.dump(vocab, dfvocab)
    with open(output_folder + '/output.pickle', 'w+') as dfoutput:
       pickle.dump(output, dfoutput)

