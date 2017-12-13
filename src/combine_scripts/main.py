from pickle import dump
from utils import load_embeddings, dim_reduce
from config import *

import os

if __name__=='__main__':
    vocab, vecs = load_embeddings(folder_in, fname, ext, nmodels, norm, arch)
    vocab, vecs = dim_reduce(vocab, vecs, ndim, mean_corr)
    d = dict(zip(vocab, vecs))
    os.system('mkdir -pv '+folder_out)
    dump(d, open(folder_out+dump_name+'.pkl', 'w+'))

