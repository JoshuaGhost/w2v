# -*- coding: utf-8 -*-
'''
    This script divide line-sentence corpus into sevral parts according to
    different strategy. The partial corpus will be saved in 
    <dir/to/parts>/<strategy_code> with name of <corpus_name>.<part_num>.txt

    Usage:
    python divide_corpus.py <strategy> <num of parts> <dir/to/temp_document> <dir/saving/partial/documents>

    For the strategy, it can be:
        skip: divide corpus so that every <num of parts> document is saved into one part
        bagging: divide corpus using bagging
'''

import logging

from gensim.models.word2vec import LineSentence
import os,sys
import os.path

def divide_skip(source, npart):
    import itertools
    step = npart
    for start in range(npart):
        fin = open(source, 'r')
        lines = fin.readlines()
        yield itertools.islice(lines, start, None, step)
        fin.close()


def divide_bagging(source, npart):
    print 'unimplemented'


if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)

    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level = logging.INFO)
    logger.info("running %s %" ' '.join(sys.argv))

    if len(sys.argv) < 5:
        print globals()['__doc__'] %locals()
        sys.exit(1)

    strategy, npart, temp_doc, parts_dir = sys.argv[1:5]
    npart = int(npart)
    corpus_name = os.path.basename(temp_doc)

    if strategy=='skip':
        part_gen = divide_skip(temp_doc, npart)
        sub_folder='skip/'
    elif strategy=='bagging':
        part_gen = divide_bagging(temp_doc, npart)
        sub_folder='bagging/'

    for i, part in enumerate(part_gen):
        filename = parts_dir+sub_folder+corpus_name+'.'+str(i)+'.txt'
        if os.path.exists(filename):
            continue
        else:
            output = open(filename, 'w+')
            for j, text in enumerate(part):
                output.write(text)
                if (j+1) % 1000 ==0:
                    logger.info('Saved '+str(j)+' articles')
            output.close()
            logger.info('finished saved partial articles No. '+str(j))

