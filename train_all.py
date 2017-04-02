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

    ftime = open(LOG_FILE, 'a+')
    ftime.write('following are times of training with corpora from %s\n'% articles_dir)
    ftime.write('trained sub-models are saved under %s\n\n'% sub_models_dir)

    for dirpath, dnames, fnames in os.walk(articles_dir):
        for midx, article_name in enumerate(fnames):
            sub_model_name = os.path.basename(article_name)+'.w2v'
            if os.path.isfile(sub_models_dir+sub_model_name):
                logger.info('sub model %s existed, skip this one' % sub_model_name)
                continue
            else:
                print article_name
                timed = -ctime()
                model = Word2Vec(LineSentence(articles_dir+article_name),
                                size = 500,
                                negative = 5,
                                workers = 18,
                                window = 10,
                                sg = 1,
                                null_word = 1,
                                min_count = new_min_count,
                                sample = 1e-4)
                timed += ctime()
                model.save(sub_models_dir+sub_model_name)
                from time import localtime, strftime
                time_now = strftime('%Y-%m-%d %H:%M:%S', localtime())
                ftime.write('training finished at %s\n' % time_now)
                ftime.write(' corpus name: %s\n'% article_name)
                ftime.write('  model name: %s\n'% sub_model_name)
                ftime.write('time elapsed: %f sec\n\n'% timed)

    ftime.write('training of individual sub-models finished, start to combine...\n')
    ftime.write('---------------------------------------\n\n')
    ftime.close()
