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
from os import path, walk
from copy import deepcopy
import pickle
import logging
import optparse
import multiprocessing

parser = optparse.OptionParser()
parser.add_option("-j", "--n_jobs", type="int", default=4)
parser.add_option("-o", "--output_dir", type="str", default="")
parser.add_option("-d", "--input_dir", type="str", default="output/models/all")
(opts, args) = parser.parse_args()

# Configure logging
logging.basicConfig(format='%(asctime)s %(levelname)s:%(message)s', level=logging.DEBUG, datefmt='%I:%M:%S')
logger = logging.getLogger(__name__)

jobs = []

for root, subdirs, files in walk(opts.input_dir):
    for fname in files:
        file_path = path.join(root, fname)
        jobs.append(['load_embedding', {'fname': file_path, 'format': 'dict'}, fname])

def run_job(j):
    fn, kwargs, model_name = j
    outf = path.join(opts.output_dir, "eval_"+model_name[:-4]) + ".csv"
    print(outf)
    logger.info("Processing " + outf)
    if not path.exists(outf):
        w = getattr(embeddings, fn)(**kwargs)
        res = evaluate_on_all(w)
        res.to_csv(outf)

if __name__ == "__main__":
    Pool(opts.n_jobs).map(run_job, jobs)
