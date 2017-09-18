from gensim.models.word2vec import Word2Vec
import sys
import numpy as np
import pickle

from encode import load_vecs, encode, dim

dump_vectors = False
non_linear = True

#(frag_folder, basename, num_models, suffix) = sys.argv[1:]
#num_models = int(num_models)
frag_folder = "temp/models/dummy"
basename = "article.txt."
suffix = ".txt.w2v"
num_models = 5

if __name__=='__main__':
    _, vec = load_vecs(frag_folder, basename,
                    suffix, num_models)

    bi = [(b, i) for b in range(10, 110, 10) for i in range(1, 23, 2)]
    bi.extend([(b, i) for b in range(200, 1100, 100) for i in range(1, 23, 2)])
    err_min = 999999999.0

    for batch, num_iter in bi:
        we, be, err = encode(vec, dim, batch, num_iter)
        if err < err_min:
            (batch_opt, iter_opt, err_min) =\
            (batch, num_iter, err)

    print ("optimized batch: %d, optimized iter: %d, minimum error: %f" %
           (batch_opt, iter_opt, err_min))

