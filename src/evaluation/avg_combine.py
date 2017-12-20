import os
import pandas as pd
from numpy import array
from web.embedding import Embedding
from web.evaluate import evaluate_on_all
from multiprocessing import Pool
from logging import Logger
from pickle import dump, load
from pathlib2 import Path
import logging

FORMAT = '%(asctime)-15s %(message)s'
logging.basicConfig(format=FORMAT)
logger = logging.getLogger('avg_evaluate')
logger.setLevel('INFO')

base_dir = Path('../../').resolve()
tmp_dir = Path('/tmp/zzj').resolve()
subs_dir = base_dir / Path('models/sampling_100/subs/')
out_fname = base_dir / Path('models/sampling_100/results/sampling_100_avg.csv')
words_dump_path =  tmp_dir/ Path('words.pkl')
vecs_dump_path = tmp_dir / Path('vecs.pkl')
subs_pattern = 'part-*'

cosine_similarity = True
nparts = 100

words = []
vecs = []

try:
    words = load(open(str(words_dump_path.resolve())))
    vecs = load(open(str(vecs_dump_path.resolve())))
except IOError:
    for idx, f in enumerate(subs_dir.glob(subs_pattern)):
        logger.info('reading file {}, current vocabulary size: {}'.format(f.name, len(words)))
        for line in f.open():
            w,v = line.strip().split(':')
            words.append(w)
            vecs.append(eval(v))
    vecs = array(vecs)
    dump(words, open(str(words_dump_path),'w+'))
    dump(vecs, open(str(vecs_dump_path),'w+'))

assert len(words) == vecs.shape[0]
assert vecs.shape[1] == 5000

size_per_sub = 5000/nparts

results_statistic = []

logger.info("embedding loading complete, {} word vectors loaded".format(len(words)))

for i in xrange(nparts):
    logger.info("evaluating embedding NO.{}".format(i))
    sub_embedding = vecs[:,i*size_per_sub:(i+1)*size_per_sub]
    assert sub_embedding.shape[0] == len(words)
    assert sub_embedding.shape[1] == 5000/nparts
    assert sub_embedding[0][0] == vecs[0][i*size_per_sub]
    d = dict(zip(words, sub_embedding))
    w = Embedding.from_dict(d)
    results = evaluate_on_all(w, cosine_similarity)
    results_statistic.append(results)

results = pd.concat(results_statistic).mean()
print results
results.to_csv(out_fname)
