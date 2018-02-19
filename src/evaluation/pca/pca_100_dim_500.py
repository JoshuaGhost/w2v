import logging
import numpy as np
import pickle
import os
import sys
import pandas as pd
import time

from sampling import reservoir
from numpy import array
from pathlib2 import Path
from numpy import array, hstack

from web.embedding import Embedding
from web.evaluate import evaluate_on_all

from utils import dim_reduce, load_embeddings

program = os.path.basename(sys.argv[0])
logger = logging.getLogger(program)
logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
logging.root.setLevel(level=logging.INFO)

debug = False 
n_subs_total = 100
cosine_similarity = False
n_subs_step = 1
n_subs_min = 9
dim_sub = 500
extension = ''
filename = 'submodel-'
n_subs_list = [i for i in xrange(n_subs_min, 11, n_subs_step)]
arch = 'submodels'
work_folder = Path('/tmp/zzj/sampling_100_dim_500/').resolve()
out_fname_pca = Path('./100_sampling_dim_500_pca_right.csv').resolve()
out_fname_concat = Path('./100_sampling_dim_500_concat_right.csv').resolve()

subs_folder = (work_folder/Path('right_subs/')).resolve()

if debug:
    vocab = ['test', 'test']
    vecs = array([[.1, .2],[.3, .4]])
    logger.info('debuging...{} {} {} {}'.format(str(work_folder), str(n_subs_list), str(out_fname_pca), str(out_fname_concat)))

results_total_pca = pd.DataFrame()
results_total_concat = pd.DataFrame()

for n_subs in n_subs_list:
    logger.info('count of sub-models to be combined: {}'.format(str(n_subs)))
    if n_subs == 9:
        n_comb = 1
    else:
        n_comb = 5
    if debug:
        logger.info('debuging, n_comb={}, n_subs={}, n_subs_list={}'.format(n_comb, n_subs, str(n_subs_list)))
    for i in xrange(n_comb):
        sub_idx_list = reservoir([sub_idx for sub_idx in range(n_subs_total-1)], n_subs)
        if debug:
            logger.info(str(sub_idx_list))
            continue
        logger.info('combination routine #{}, combining {}'.format(i, sub_idx_list))
        vocab, vecs_merge = load_embeddings(str(subs_folder), filename, '', True, 'submodels', selected=sub_idx_list)
        results = evaluate_on_all(dict(zip(vocab, vecs_merge)), cosine_similarity)
        results['nsubs'] = n_subs
        logger.info('evaluation results for concatenating {}:\n{}'.format(sub_idx_list, results))
        results_total_concat = results_total_concat.append(results)
        results_total_concat.to_csv(str(out_fname_concat))

        logger.info('sub embeddings concatenated, start to reduce dimensions...')
        if dim_sub == 500:
            eigval_threshold = 2.
        else:
            eigval_threshold = 2.
        telapse = -time.time()
        vecs_merge = dim_reduce(vecs_merge, eigval_threshold, True)
        telapse += time.time()
        logger.info('{} sub models merged, new dim: {}'.format(n_subs, vecs_merge.shape[1]))
        results = evaluate_on_all(dict(zip(vocab, vecs_merge)), cosine_similarity)
        results['nsubs'] = n_subs
        if n_subs == 10:
            results['time(sec)'] = telapse
        logger.info('evaluation results for merging {}:\n{}'.format(sub_idx_list, results))
        results_total_pca = results_total_pca.append(results)
        results_total_pca.to_csv(str(out_fname_pca))
        vocab = None
        vecs_merge = None

    print results_total_pca
    print results_total_concat
    results_total_pca.to_csv(str(out_fname_pca))
    results_total_concat.to_csv(str(out_fname_concat))

