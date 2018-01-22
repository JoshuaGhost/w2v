import logging
import os
import sys
import pandas as pd

from pathlib2 import Path
from numpy import hstack
from multiprocessing import Pool

from web.embedding import Embedding
from web.evaluate import evaluate_on_all

from sampling import reservoir
from utils import dim_reduce, load_embeddings

program = os.path.basename(sys.argv[0])
logger = logging.getLogger(program)
logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
logging.root.setLevel(level=logging.INFO)

debug = False 
cosine_similarity = False

n_subs_total = 100
n_subs_step = 10
n_subs_min = 2
dim_sub = 50
dim_merge = 50
n_comb = 5

if debug:
    n_subs_step = 90
    n_comb = 1

work_folder = Path('../../../models/sampling_10/').resolve()
work_folder = Path('../../../models/sampling_100/').resolve()
subs_folder = (work_folder/Path('subs')).resolve()
filename = 'sub_'
filename = 'part-'
extension = ''

out_fname = Path('./sampling_1_100th_concat.csv').resolve()

logger.info('start to collect embeddings')
vocab, vecs = load_embeddings(folder=str(subs_folder), filename=filename, extension=extension, norm=True, arch='mapreduce')
logger.info('embeddings collected, in total {} word vectors loaded'.format(len(vecs)))

def worker(argvs):
    worker_idx, idx_list = argvs
    logger.info('worker #{}: combining {}'.format(worker_idx, idx_list))
    vecs_merge = hstack(vecs[:,sub_idx*dim_sub:(sub_idx+1)*dim_sub] for sub_idx in idx_list)
    logger.info('worker #{}: {} sub models merged, new dim: {}'.format(worker_idx, len(idx_list), vecs_merge.shape[1]))
    d = dict(zip(vocab, vecs_merge))
    w = Embedding.from_dict(d)
    results = evaluate_on_all(w, cosine_similarity)
    results['nsubs'] = len(idx_list)
    logger.info("worker #{}: evaluation finished".format(worker_idx))
    return results

pool = Pool(min(n_comb, 18))

if __name__=='__main__':
    results_total = pd.DataFrame()

    n_subs_list = [i for i in xrange(n_subs_min, n_subs_total+1, n_subs_step)]
    n_subs_list = [1,] + n_subs_list
    n_subs_list = [1,]+[i for i in xrange(2, 10, 2)]
    n_subs_list += [i for i in xrange(10, 101, 10)]

    for n_subs in n_subs_list:
        logger.info('count of sub-models to be combined: {}'.format(n_subs))
        if n_subs == n_subs_total:
            n_comb = 1
        idx_lists = [reservoir([idx for idx in range(n_subs_total)], n_subs) for i in xrange(n_comb)]
        argvs_list = zip(xrange(n_comb), idx_lists)
        results = pool.map(worker, argvs_list)
        results = pd.concat(results)
        print results
        results_total = results_total.append(results)
        results_total.to_csv(str(out_fname))

    print results_total
    results_total.to_csv(str(out_fname))

