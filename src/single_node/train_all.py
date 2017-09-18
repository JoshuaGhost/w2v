# -*- coding: utf-8 -*-
"""
    This script train each of the article file under <dir/to/temp_document> into
    seprated model saved under <dir/to/sub_models>.

    Usage:
    python train_all.py <dir/to/temp_document> <dir/to/sub_models>
"""
import logging

from gensim.models.word2vec import Word2Vec
from gensim.models.word2vec import LineSentence

import os, sys
import os.path

from os import listdir
from os.path import isfile, join
from time import time as ctime

from config import LOG_FILE

if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)

    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level = logging.INFO)
    logger.info("running %s" % ' '.join(sys.argv))

    if len(sys.argv) < 3:
        print globals()['__doc__'] % locals()
        sys.exit(1)

    articles_dir, sub_models_dir = sys.argv[1:3]

    num_sub_models =  len([name for name in listdir(articles_dir) if isfile(join(articles_dir, name))])
    new_min_count = 100/num_sub_models

    logger.info('Note: corpora are from %s, Sub-models are saved under %s' % (articles_dir, sub_models_dir))

    for dirpath, dnames, fnames in os.walk(articles_dir):
        for midx, article_name in enumerate(fnames):
            sub_model_name = '.'.join(os.path.basename(article_name).split('.')[:-1])+'.w2v'
            if os.path.isfile(sub_models_dir+sub_model_name):
                logger.info('sub model %s existed, skip this one' % sub_model_name)
                continue
            else:
                logger.info('Start to train sub-model %s' % sub_model_name)
                model = Word2Vec(LineSentence(articles_dir+article_name),
                                size = 500,
                                negative = 5,
                                workers = 18,
                                window = 10,
                                sg = 1,
                                null_word = 1,
                                min_count = new_min_count,
                                sample = 1e-4)
                logger.info('Sub-model %s train finished' % sub_model_name)
                model.save(sub_models_dir+sub_model_name)

    logger.info('Sub-models training finished.')
