import logging
import numpy as np
import pickle
import os
import sys
import pandas as pd
import random

from lra import low_rank_align
from numpy import array
from pathlib2 import Path
from numpy import array

from web.embedding import Embedding
from web.evaluate import evaluate_on_all
from utils import load_embeddings

program = os.path.basename(sys.argv[0])
logger = logging.getLogger(program)
logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
logging.root.setLevel(level=logging.INFO)
work_folder = Path('../../mapreduce/evaluation/').resolve()
subs_folder = (work_folder/Path('submodels/')).resolve()
filename = 'submodel-'
extension = ''
fvocab = './vocab_bench.txt'
out_fname = Path('./lra_smaller_corpus.csv').resolve()

if __name__=='__main__':
    debug = False
    
    cosine_similarity = True

    n_subs_total = 10
    n_subs_step = 1
    n_subs_min = 1
    dim_sub = 50
    dim_merge = 50
    n_comb = 5

    if debug:
        n_subs_step = 90
        n_comb = 1

    vocab = [] 
    word2idx = {}
    vecs = []
    
    logger.info('start to collect embeddings')
    vocab_bench = set(open(fvocab).read().split())
    logger.info('embeddings collected, in total {} word vectors loaded'.format(str(len(vecs))))

    results_total = pd.DataFrame()

    n_subs_list = [i for i in range(1, 11)]

    for i in xrange(10):
        idx_list = random.sample([idx for idx in range(n_subs_total)], 10)
        logger.info('combination routine #{}, combining {}'.format(str(i), str(idx_list)))
        if debug:
            vocab = ['a', 'abandon']
            vecs_merge = [[1., 2.], [3., 9.]]
        else:
            vocab, vecs_merge = load_embeddings(folder=str(subs_folder),
                                                filename=filename,
                                                extension=extension,
                                                norm=False,
                                                arch='submodels',
                                                selected=[idx_list[0],])
        vecs_merge = [vecs_merge[idx] for idx, word in enumerate(vocab) if word in vocab_bench]
        vocab = [word for word in vocab if word in vocab_bench]
        vecs_merge = array(vecs_merge)

        for nmerged, sub_idx in enumerate(idx_list[1:]):
            if debug:
                vo = ['a', 'abandon']
                vecs = [[2., 4.], [3., 4.]]
            else:
                vo, vecs = load_embeddings(folder=str(subs_folder),
                                            filename=filename,
                                            extension=extension,
                                            norm=False,
                                            arch='submodels',
                                            selected=[sub_idx, ])
            vecs = [vecs[idx] for idx, word in enumerate(vo) if word in vocab]
            vecs = array(vecs)
            Fx, Fy = low_rank_align(vecs_merge, vecs, np.eye(vecs.shape[0]))
            vecs_merge = (Fx*(nmerged + 1)+Fy)/(nmerged + 2)
            d = dict(zip(vocab, vecs_merge))
            w = Embedding.from_dict(d)
            if debug:
                results = pd.DataFrame([[1,2]], columns=['a', 'b'])
            else:
                results = evaluate_on_all(w, cosine_similarity)
            results['nsubs'] = nmerged + 2
            results_total = results_total.append(results)
            results_total.to_csv(str(out_fname))
            logger.info('{} sub models merged, results: {}'.format(nmerged+2, results))

    print results_total
    results_total.to_csv(str(out_fname))

