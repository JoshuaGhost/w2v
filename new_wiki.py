# -*- coding: utf-8 -*-
"""
    This script build a vector representation from wiki dump copora 
    and saves in w2v model file

    Usage:
    python new_wiki.py ../corpus/enwiki-latest-pages-articles.xml.bz2 ./temp_article/article.txt ./temp_article/new_wiki.w2v
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

    if len(sys.argv) < 4:
        print globals()['__doc__'] % locals()
        sys.exit(1)

    wiki_dump_name, article_name, model_name = sys.argv[1:4]
    space = " "

    if os.path.isfile(article_name):
        logger.info('Article file exists, using existing article file')
    else:
        i = 0
        output = open(article_name, 'w')
        wiki = WikiCorpus(wiki_dump_name, lemmatize = False, dictionary = {})
        for text in wiki.get_texts():
            output.write(space.join(text) + '\n')
            i += 1
            if (i % 1000 == 0):
                logger.info("Saved " + str(i) + " articles")
        output.close()
        logger.info("Finished Saved " + str(i) + " articles")

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
        logger.info('model file exists, will not cover it')
    else:
        model.save_word2vec_format(model_name, binary = False)

    from time import localtime, strftime
    times = strftime("%Y-%m-%d %H:%M:%S", localtime())

    with open('result/default_results/new_wiki_time.txt', 'a+') as f:
        f.write("\nfinished at %s, training time of new_wiki: %f sec\n" % (times, timed))
