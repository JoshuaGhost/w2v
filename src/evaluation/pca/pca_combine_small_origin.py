import logging
import numpy as np
import pickle
import os
import sys
import pandas as pd
import random

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
n_subs_total = 10
n_subs_max = 10
cosine_similarity = False
n_subs_step = 1
n_subs_min = 1
dim_sub = 50
n_comb = 5
extension = ''
eigval_threshold = 1.
filename = 'submodel-'
n_subs_list = [i for i in xrange(n_subs_min, n_subs_total+1, n_subs_step)]
arch = 'submodels'
work_folder = Path('../submodels/').resolve()
out_fname_pca = Path('./10_sampling_dim_50_smaller_origin_pca.csv').resolve()
out_fname_concat = Path('./10_sampling_dim_50_smaller_origin_concat.csv').resolve()
subs_folder = (work_folder/Path('submodels/')).resolve()

if debug:
    logger.info('debuging...{} {} {} {}'.format(str(work_folder), str(n_subs_list), str(out_fname_pca), str(out_fname_concat)))

results_total_pca = pd.DataFrame()
results_total_concat = pd.DataFrame()

for i in xrange(n_comb):
    if i == 0:
        n_subs_max = n_subs_total
    else:
        n_subs_max = n_subs_total - n_subs_step
    sub_idx_list = random.sample([sub_idx for sub_idx in range(n_subs_total)], n_subs_max)
    
    if debug:
        logger.info('debuging, n_comb={}, n_subs={}, n_subs_list={}'.format(n_comb, n_subs, str(n_subs_list)))
        continue

    vocab, vecs = load_embeddings(str(subs_folder), filename, '', True, 'submodels', selected=sub_idx_list)
    vecs_merge = [[] for word in vocab]
    for idx, sub_idx in enumerate(sub_idx_list):

        vecs_merge = np.hstack((vecs_merge, vecs[:, sub_idx*dim_sub:(sub_idx+1)*dim_sub]))
        logger.info('combination routine #{}, combining {}'.format(i, sub_idx_list[:idx+1]))
        results = evaluate_on_all(dict(zip(vocab, vecs_merge)), cosine_similarity)
        results['nsubs'] = idx+1
        logger.info('evaluation results for concatenating {}:\n{}'.format(sub_idx_list[:idx+1], results))
        results_total_concat = results_total_concat.append(results)
        results_total_concat.to_csv(str(out_fname_concat))

        logger.info('sub embeddings concatenated, start to reduce dimensions...')

        vecs_merge = dim_reduce(vecs_merge, eigval_threshold, True)
        logger.info('{} sub models merged, new dim: {}'.format(idx+1, vecs_merge.shape[1]))
        results = evaluate_on_all(dict(zip(vocab, vecs_merge)), cosine_similarity)
        results['nsubs'] = idx+1
        logger.info('evaluation results for merging {}:\n{}'.format(sub_idx_list[:idx+1], results))
        results_total_pca = results_total_pca.append(results)
        results_total_pca.to_csv(str(out_fname_pca))

    print results_total_pca
    print results_total_concat
    results_total_pca.to_csv(str(out_fname_pca))
    results_total_concat.to_csv(str(out_fname_concat))

