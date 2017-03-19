# -*- coding: utf-8 -*-
"""
    This script builds the base_line for the whole experiment from a 
    dumped wiki corpora and saves in w2v model file

    Usage:
    python base_line.py <dir/to/temp_document> <dir/to/model_file>
"""

import logging

from gensim.models.word2vec import Word2Vec
from gensim.models.word2vec import LineSentence
from gensim.corpora import WikiCorpus
import os, sys
from time import time as ctime
import os.path

if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)

    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level = logging.INFO)
    logger.info("running %s %" ' '.join(sys.argv))

    if len(sys.argv) < 3:
        print globals()['__doc__'] % locals()
        sys.exit(1)

    article_name, model_name = sys.argv[1:3]
    space = " "

    timed = -ctime()
    model = Word2Vec(LineSentence(article_name),
                     size = 500,
                     negative = 5,
                     workers = 18,
                     window = 10,
                     sg = 1,
                     null_word = 1,
                     min_count = 100,
                     sample = 1e-4)
    timed += ctime()
    if os.path.isfile(model_name):
        logger.info('model file exists, will not overwrite it')
    else:
        model.save(model_name)

    from time import localtime, strftime
    times = strftime("%Y-%m-%d %H:%M:%S", localtime())

    with open('result/default_results/new_wiki_time.txt', 'a+') as f:
        f.write("finished at %s, training time of new_wiki: %f sec\n\n" % (times, timed))
