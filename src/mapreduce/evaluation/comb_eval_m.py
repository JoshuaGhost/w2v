#!./env/bin/python

import logging
import numpy as np
import pickle
import os
import pandas as pd

from numpy import array
from pathlib2 import Path
from numpy import array, hstack

import sys
sys.path.append('./')

from mine.sampling import reservoir
from web.embedding import Embedding
from web.evaluate import evaluate_on_all
from mine.utils import dim_reduce, load_embeddings

program = os.path.basename(sys.argv[0])
logger = logging.getLogger(program)
logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
logging.root.setLevel(level=logging.INFO)

subs_folder = 'submodels/'
filename = 'submodel-'
extension = ''
cosine_similarity = False

for line in sys.stdin:
    idx_list = eval(line)
    n_subs = len(idx_list)
    vocab, vecs_merge = load_embeddings(subs_folder, filename, extension, True, 'submodels', selected=idx_list)
    logger.info('sub modes:{} collected and concatenated, start to evaluate concatenation'.format(idx_list))
    results_concat = evaluate_on_all(dict(zip(vocab, vecs_merge)), cosine_similarity)
    results_concat['nsubs'] = n_subs
    logger.info('evaluation results for concatenation {}:\n{}'.format(idx_list, results_concat))

    eigval_threshold = 2.
    vecs_merge = dim_reduce(vecs_merge, eigval_threshold, True)
    logger.info('{} sub models merged, new dim: {}'.format(n_subs, vecs_merge.shape[1]))
    results_pca = evaluate_on_all(dict(zip(vocab, vecs_merge)), cosine_similarity)
    results_pca['nsubs'] = n_subs
    logger.info('evaluation results for pca {}:\n{}'.format(idx_list, results_pca))

    print str(n_subs)+'\t'+repr({'concat':results_concat.to_dict(), 'pca':results_pca.to_dict()})

