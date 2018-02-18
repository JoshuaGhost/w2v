#!./env/bin/python
from itertools import islice
from math import log
import sys
import copy, random
import os
from pathlib2 import Path
from sampling import reservoir
import logging

original_corpus_folder = Path('../subcorpora/').resolve()
original_corpus_path = original_corpus_folder/Path('part.article.0.txt')
sub_corpora_folder = Path('./').resolve()
nsub_corpora = 10
sub_name = 'part.article'
sub_ext = 'txt'
strategy = 'sampling'
#nlines_origin = int(os.popen('wc '+str(original_corpus_path)).read().split()[0])
nlines_origin = 422794

FORMAT = '%(asctime)-15s %(message)s'
logging.basicConfig(format=FORMAT)
logger = logging.getLogger('divider')
logger.setLevel(logging.INFO)

if __name__ == '__main__':
    size_in_byte_origin = os.stat(str(original_corpus_path)).st_size
    nlines_sub_corpus = nlines_origin/nsub_corpora+1
    size_in_byte_sub = size_in_byte_origin/nsub_corpora+1

    logger.info('start to split original corpus into {} sub-corpora'.format(nsub_corpora))

    origin = open(str(original_corpus_path), 'r')
    for idx in range(nsub_corpora):
        conf = [strategy, sub_name, str(idx).zfill(int(log(nsub_corpora)/log(10)+1)), sub_ext]
        sub_corpus_path = sub_corpora_folder/Path('.'.join(conf))
        fsub = open(str(sub_corpus_path), 'w+')
        if strategy == 'skip':
            origin.seek(0)
            sub_corpus = [s for s in islice(origin.readlines(), idx, None, nsub_corpora)]
        elif strategy == 'sampling':
            origin.seek(0)
            sub_corpus = [s for s in reservoir(origin.readlines(), nlines_sub_corpus)]
        elif strategy == 'chunk':
            sub_corpus = origin.read(size_in_byte_sub)
        fsub.writelines(sub_corpus)
        fsub.close()
        logger.info('sub corpus No. {} complete'.format(idx))
    origin.close()

