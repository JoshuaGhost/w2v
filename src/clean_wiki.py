# -*- coding: utf-8 -*-
'''
    This is the script generate line-scentence wiki corpora for gensim, using
    latest wiki dump. The cleaned documents will be saved as article.txt under
    <dir/saving/temp_documents>
    
    Usage:
    python clean_wiki.py <dir/to/corpus_file> <dir/saving/temp_documents>
'''

import os, sys
from time import time as ctime
import os.path

from config import *

if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)

    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level = logging.INFO)
    logger.info("running %s %" ' '.join(sys.argv))

    if len(sys.argv) < 3:
        print globals()['__doc__'] % locals()
        sys.exit(1)

    wiki_corpus_file, article_name= sys.argv[1:3]
    space = " "

    if os.path.isfile(article_name):
        logger.info('Article file exists, will not overwrite existed article file')
    else:
        i = 0
        output = open(article_name, 'w+')
        wiki = WikiCorpus(wiki_corpus_file, lemmatize = False, dictionary = {})
        for text in wiki.get_texts():
            output.write(space.join(text) + '\n')
            i += 1
            if (i % 1000 == 0):
                logger.info("Saved " + str(i) + " articles")
        output.close()
        logger.info("Finished Saved " + str(i) + " articles")

