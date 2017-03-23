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
from time import time as ctime

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
    logger.info("running %s" % ' '.join(sys.argv))

    if len(sys.argv) < 5:
        print globals()['__doc__'] %locals()
        sys.exit(1)
    
    strategy, npart, temp_doc, parts_dir = sys.argv[1:5]
    npart = int(npart)
    corpus_name = os.path.basename(temp_doc)

    from time import localtime, strftime
    times = strftime("%Y-%m-%d %H:%M:%S", localtime())

    ftime = open('result/default_results/time_alignment_combine.txt', 'a+')
    ftime.write('=======================================\n')
    ftime.write("experiment started at %s\n" % times)
    ftime.write('running %s\n' % ' '.join(sys.argv))
    ftime.write('divide corpus file [%s] into [%d] part with strategy [%s]\n' % (temp_doc, npart, strategy))
    ftime.write('divided corpus saved under [%s]\n' % parts_dir)

    if strategy=='skip':
        part_gen = divide_skip(temp_doc, npart)
        sub_folder='skip/'
    elif strategy=='bagging':
        part_gen = divide_bagging(temp_doc, npart)
        sub_folder='bagging/'

    for i, part in enumerate(part_gen):
        filename = parts_dir+sub_folder+corpus_name+'.'+str(i)+'.txt'
        etime = -ctime()
        if os.path.exists(filename):
            continue
        else:
            output = open(filename, 'w+')
            for j, text in enumerate(part):
                output.write(text)
                if (j+1) % 1000 ==0:
                    logger.info('Saved '+str(j)+' articles')
            output.close()
            etime += ctime()
            ftime.write('part %d finished, time elapsed: %f sec\n' % (j, etime))
            logger.info('finished saved partial articles No. '+str(i))

    ftime.write('division finished, start to train...\n')
    ftime.write('---------------------------------------\n\n')
    ftime.close()
