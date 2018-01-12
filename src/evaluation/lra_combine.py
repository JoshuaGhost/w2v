import logging
import numpy as np
import pickle
import os
import sys
import pandas as pd

from lra import low_rank_align
from sampling import reservoir
from numpy import array
from pathlib2 import Path
from numpy import array

from web.embedding import Embedding
from web.evaluate import evaluate_on_all

program = os.path.basename(sys.argv[0])
logger = logging.getLogger(program)
logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
logging.root.setLevel(level=logging.INFO)

fvocab = './vocab_bench.txt'
out_fname = Path('./lra_comb_eva.csv').resolve()

class Selector:
    def __init__(self, n, k, init=None):
        self.n = n
        self.k = k
        if init is not None:
            self.output = init
        else:
            self.output = [i for i in range(k)]
        self.output[-1]-=1
        assert len(self.output) == k

    def find_next(self, p):
        if p == self.k:
            return -1
        if self.output[p] == self.n-(self.k-p):
            return -1
        ret = self.find_next(p+1)
        if ret == -1:
            self.output[p:] = [self.output[p]+i+1 for i, c in enumerate(self.output[p:])]
            return 1
        return ret
    
    def __iter__(self):
        return self

    def next(self):
        ret = self.find_next(0)
        if ret == -1:
            raise StopIteration
        return self.output

if __name__=='__main__':
    debug = False
    
    cosine_similarity = True

    n_subs_total = 100
    n_subs_step = 10
    n_subs_min = 10
    dim_sub = 50
    dim_merge = 50
    n_comb = 5

    if debug:
        n_subs_step = 90
        n_comb = 1

    work_folder = Path('../../models/sampling_100/').resolve()
    subs_folder = (work_folder/Path('subs')).resolve()
    vocab = [] 
    word2idx = {}
    vecs = []
    
    logger.info('start to collect embeddings')
    vocab_bench = set(open(fvocab).read().split())
    for part_name in subs_folder.glob('part-*'):
        for line in open(str(part_name)):
            word, vec = line.split(':')
            word = eval(word)
            vec = eval(vec)
            if word in vocab_bench:
                word2idx[word] = len(vecs)
                vecs.append(vec)
                vocab.append(word)
        if debug == True:
            break
    
    vecs = array(vecs)
    logger.info('embeddings collected, in total {} word vectors loaded'.format(str(len(vecs))))

    results_total = pd.DataFrame()

    n_subs_list = [i for i in xrange(n_subs_min, n_subs_total, n_subs_step)]
    n_subs_list = [1,]+n_subs_list

    for n_subs in n_subs_list:
        logger.info('count of sub-models to be combined: {}'.format(str(n_subs)))
        if n_subs == n_subs_total:
            n_comb = 1
        for i in xrange(n_comb):
            idx_list = reservoir([idx for idx in range(n_subs_total)], n_subs)
            logger.info('combination routine #{}, combining {}'.format(str(i), str(idx_list)))
            vecs_merge = vecs[:,:dim_sub]
            for nmerged, sub_idx in enumerate(idx_list[1:]):
                Fx, Fy = low_rank_align(vecs_merge, vecs[:, sub_idx*dim_sub:(sub_idx+1)*dim_sub], np.eye(vecs.shape[0]))
                vecs_merge = (Fx*(nmerged + 1)+Fy)/(nmerged + 2)
                logger.info('{} sub models merged'.format(nmerged+2))
            d = dict(zip(vocab, vecs_merge))
            w = Embedding.from_dict(d)
            results = evaluate_on_all(w, cosine_similarity)
            results['nsubs'] = n_subs
            print results
            results_total = results_total.append(results)
            results_total.to_csv(str(out_fname))

    print results_total
    results_total.to_csv(str(out_fname))

