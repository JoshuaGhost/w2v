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

from utils import dim_reduce, load_embeddings

program = os.path.basename(sys.argv[0])
logger = logging.getLogger(program)
logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
logging.root.setLevel(level=logging.INFO)

for n_subs_total in [10, 100]:
    debug = False 
    if n_subs_total == 10:
        dim_sub = 500
        dim_merge = 500
        extension = ''
        filename = 'sub_'
        arch = 'local'
        work_folder = Path('/tmp/zzj/sampling_10/').resolve()
        out_fname = Path('/tmp/zzj/sampling_10_pca.pkl').resolve()
    else:
        dim_sub = 50
        dim_merge = 50
        extension = ''
        filename = 'part-'
        arch = 'mapreduce'
        work_folder = Path('../../models/sampling_100/').resolve()
        out_fname = Path('/tmp/zzj/sampling_100_pca.pkl').resolve()

    dim_sub = 500
    filename = 'part-'
    extension = ''
    arch = 'mapreduce'
    work_folder = Path('/tmp/zzj/sampling_100_dim_500').resolve()
    out_fname = Path('/tmp/zzj/sampling_100_dim_500_pca.pkl').resolve()

    subs_folder = (work_folder/Path('subs/')).resolve()

    subs_folder = work_folder

    logger.info('start to collect embeddings')
    if debug:
        vocab = ['test', 'test']
        vecs = array([[.1, .2],[.3, .4]])
        logger.info('debuging...{} {}'.format(str(work_folder), str(out_fname)))
    else:
        vocab, vecs = load_embeddings(folder=str(subs_folder), filename=filename, extension=extension, norm=True, arch=arch)

    logger.info('embeddings collected, in total {} word vectors loaded'.format(str(len(vecs))))

    logger.info('start to reduce dimensions...')
    if dim_sub == 500:
        eigval_threshold = 10.
    else:
        eigval_threshold = 2.
    vecs_merge = dim_reduce(vecs, eigval_threshold, True)
    logger.info('{} sub models merged, new dim: {}'.format(n_subs_total, vecs_merge.shape[1]))
    pickle.dump(dict(zip(vocab, vecs_merge)), open(str(out_fname), 'w+'))
