# -*- coding: utf-8 -*-
'''
    This script divide line-sentence corpus into sevral parts according to
    different strategy. The partial corpus will be saved in 
    <dir/to/parts>/<strategy_code> with name of <corpus_name>.<part_num>.txt

    Usage:
    python divide_corpus.py <strategy> <num of parts> <dir/to/corpus_fileument> <dir/saving/partial/documents>

    For the strategy, it can be:
        partitioning: divide corpus so that every <num of parts> document is saved into one part
        sampling: divide corpus using sampling
'''

import logging
import os,sys
import os.path

from gensim.models.word2vec import LineSentence

from time import time as ctime

from scipy.sparse import lil_matrix

from random import randint

from config import LOG_FILE


def sample_table(ntotal, nparts):
    num_per_part = ntotal/nparts
    i = 0
    ret = lil_matrix((1,ntotal))
    while i<num_per_part:
        while True:
            p = randint(0, ntotal-1)
            if ret[0, p] == 0:
                ret[0, p] = 1
                break
        i += 1
    return ret


def part_wrt_hit_table(source, hit):
    source.seek(0)
    for i, line in enumerate(source.readlines()):
        if hit[0, i] != 0:
            yield line


def divide_partitioning(source, nparts):
    import itertools
    step = nparts
    for start in range(nparts):
        lines = source.readlines()
        yield itertools.islice(lines, start, None, step)


def divide_sampling(source, nparts):
    source.seek(0)
    nline = sum(1 for line in source)
    for i in range(nparts):
        source.seek(0)
        hit = sample_table(nline, nparts)
        yield part_wrt_hit_table(source, hit)


class DividedLineSentence(object):

    def __init__(self, strategy, npart, source):
        self.strategy = strategy
        self.npart = npart
        self.source_doc = open(source, 'r')
        if strategy=='partitioning':
            self.all_parts = divide_partitioning(self.source_doc, self.npart)
            self.sub_folder='partitioning/'
        elif strategy=='sampling':
            self.all_parts = divide_sampling(self.source_doc, self.npart)
            self.sub_folder='sampling/'

    def __iter__(self):
        self.source_doc.seek(0)
        for part in self.all_parts:
            yield part
            
    def __del__(self):
        self.source_doc.close()

        
if __name__ == '__main__':
    program = os.path.basename(sys.argv[0])
    logger = logging.getLogger(program)

    logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
    logging.root.setLevel(level = logging.INFO)
    logger.info("running %s" % ' '.join(sys.argv))

    if len(sys.argv) < 5:
        print globals()['__doc__'] %locals()
        sys.exit(1)
    
    strategy, npart, corpus_file, parts_dir = sys.argv[1:5] #strategy = {sampling, partitioning}
    npart = int(npart)
    corpus_name = '.'.join(os.path.basename(corpus_file).split('.')[:-1])

    logger.info('start to divide corpora')
    
    from time import localtime, strftime

    logger.info('divide corpus file [%s] into [%d] part with strategy [%s]' % (corpus_file, npart, strategy))
    logger.info('divided corpus saved under [%s]' % parts_dir)

    lineSentences = DividedLineSentence(strategy, npart, corpus_file)
    for idx, part in enumerate(lineSentences):
        sub_folder = lineSentences.sub_folder
        filename = parts_dir+'/'+sub_folder+'/'+corpus_name+'.'+str(idx)+'.txt'
        if os.path.exists(filename):
            logger.info('part %d exists, skipped' % idx)
            continue
        else:
            output = open(filename, 'w+')
            for j, text in enumerate(part):
                output.write(text)
                if (j+1) % 1000 ==0:
                    logger.info('Saved '+str(j)+' articles')
            output.close()
            logger.info('part %d generated' % idx)
            logger.info('finished saving partial articles No. '+str(idx))

