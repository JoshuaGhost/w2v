import logging
import os
import sys
import pandas as pd
import pickle

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
dim_sub = 50
n_comb = 6

if debug:
    n_subs_step = 90
    n_comb = 1

work_folder = Path('../../../models/sampling_100/').resolve()
subs_folder = (work_folder/Path('subs')).resolve()
filename = 'part-'
extension = ''

out_fname = Path('./sampling_1_100th_concat_correct_part3.csv').resolve()

logger.info('start to collect embeddings')
try:
    vocab = pickle.load(open('vocab_1_100th.pkl'))
    vecs = pickle.load(open('vecs_1_100th.pkl'))
except IOError:
    vocab, vecs = load_embeddings(folder=str(subs_folder), filename=filename, extension=extension, norm=True, arch='mapreduce')
    pickle.dump(vocab, open('vocab_1_100th.pkl', 'w+'))
    pickle.dump(vecs, open('vecs_1_100th.pkl', 'w+'))

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

    n_subs_list = [100, 70, 50, 20, 10, 5]
    for n_subs in n_subs_list:
        n_comb = 6
        if n_subs == 100:
            n_comb = 1
        logger.info('count of sub-models to be combined: {}'.format(n_subs))
        idx_lists = [reservoir([idx for idx in range(n_subs_total)], n_subs) for i in xrange(n_comb)]
        argvs_list = zip(xrange(n_comb), idx_lists)
        results = pool.map(worker, argvs_list)
        results = pd.concat(results)
        print results
        results_total = results_total.append(results)
        results_total.to_csv(str(out_fname))

    print results_total
    results_total.to_csv(str(out_fname))

