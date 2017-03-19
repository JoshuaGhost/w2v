#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
 This script evaluates all embeddings available in the package
 and saves .csv results

 Usage:

 ./evaluate_embeddings <output_dir>
"""
from web.evaluate import evaluate_on_all
from web import embeddings
from six import iteritems
from multiprocessing import Pool
from os import path
import logging
import optparse
import multiprocessing

parser = optparse.OptionParser()
parser.add_option("-j", "--n_jobs", type="int", default=4)
parser.add_option("-o", "--output_dir", type="str", default="")
(opts, args) = parser.parse_args()

# Configure logging
logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', level=logging.DEBUG, datefmt='%I:%M:%S')
logger = logging.getLogger(__name__)

jobs = []

for mc in range(0, 49, 2):
    for dim in range(25, 201, 25):
        model_dir = "/home/assassin/workspace/master_thesis/models/new/"
        model_name = "dim_%d_mc_%d_iter.w2v"%(dim, mc)
        model_path = model_dir + model_name
        jobs.append(['from_gensim', {"fname": model_path, "dim": dim, "corpus":"wiki_news", "min_count": mc}])

def run_job(j):
    fn, kwargs = j
    outf = path.join(opts.output_dir, fn + "_" + "_".join(str(k) + "=" + str(v) for k,v in iteritems(kwargs) if not k == 'fname')) + ".csv"
    print(outf)
    logger.info("Processing " + outf)
    if not path.exists(outf):
        w = getattr(embeddings, fn)(**kwargs)
        res = evaluate_on_all(w)
        res.to_csv(outf)

if __name__ == "__main__":
    Pool(opts.n_jobs).map(run_job, jobs)
