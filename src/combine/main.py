from pickle import dump
from utils import load_embeddings, dim_reduce
from config import *

import os

if __name__=='__main__':
    vocab, vecs = load_embeddings(folder_in, fname, ext, norm, arch)
    if combine_method == 'pca':
        vocab, vecs = dim_reduce(vocab, vecs, ndim, mean_corr)
    elif combine_method == 'concate':
        pass
    d = dict(zip(vocab, vecs))
    os.system('mkdir -pv '+folder_out)
    dump(d, open(folder_out+dump_name+'.pkl', 'w+'))

