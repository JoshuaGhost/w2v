import logging
import numpy as np
import pickle
import os
import sys
import pandas as pd

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

out_fname = Path('./pca_comb_eva_1_to_10.csv').resolve()
out_fname = Path('./pca_sampling_1_of_10th_1_to_10.csv').resolve()
work_folder = Path('../../models/sampling_100/').resolve()
work_folder = Path('/tmp/zzj/sampling_10/').resolve()
subs_folder = (work_folder/Path('subs/')).resolve()
subs_folder = (work_folder/Path('subs/')).resolve()

if __name__=='__main__':
    debug = False 
   
    cosine_similarity = False
    n_subs_total = 10
    n_subs_step = 2
    n_subs_min = 2
    dim_sub = 500
    dim_merge = 500
    extension = ''
    filename = 'sub_'

    if debug:
        extension = ''
   
    logger.info('start to collect embeddings')
    vocab, vecs = load_embeddings(folder=str(subs_folder), filename=filename, extension=extension, norm=True, arch='local')
    logger.info('embeddings collected, in total {} word vectors loaded'.format(str(len(vecs))))

    results_total = pd.DataFrame()
    n_subs_list = [i for i in xrange(n_subs_min, n_subs_total+1, n_subs_step)]
    n_subs_list = [1,]+n_subs_list

    for n_subs in n_subs_list:
        logger.info('count of sub-models to be combined: {}'.format(str(n_subs)))
        n_comb = 10
        if n_subs == n_subs_total:
            n_comb = 1
        if debug:
            logger.info('debuging, n_comb={}, n_subs={}, n_subs_list={}'.format(n_comb, n_subs, str(n_subs_list)))
            continue
        for i in xrange(n_comb):
            sub_idx_list = reservoir([sub_idx for sub_idx in range(n_subs_total)], n_subs)
            logger.info('combination routine #{}, combining {}'.format(str(i), str(sub_idx_list)))
            vecs_merge = hstack([vecs[:,sub_idx*dim_sub:(sub_idx+1)*dim_sub] for sub_idx in sub_idx_list])
            logger.info('sub embeddings concatenated, start to reduce dimensions...')
            _, vecs_merge = dim_reduce(None, vecs_merge, dim_merge, True)
            logger.info('{} sub models merged, new dim: {}'.format(str(n_subs), str(dim_merge)))
            w = Embedding.from_dict(dict(zip(vocab, vecs_merge)))
            results = evaluate_on_all(w, cosine_similarity)
            results['nsubs'] = n_subs
            logger.info('evaluation results for merging {}:\n{}'.format(str(sub_idx_list), str(results)))
            results_total = results_total.append(results)
            results_total.to_csv(str(out_fname))

    print results_total
    results_total.to_csv(str(out_fname))

