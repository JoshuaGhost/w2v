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

def part_wrt_hit_table(lines, hit):
    for i, line in enumerate(lines):
        if hit[0, i] != 0:
            yield line

def divide_partitioning(source, nparts):
    import itertools
    step = nparts
    for start in range(nparts):
        lines = source.readlines()
        yield itertools.islice(lines, start, None, step)

def divide_sampling(source, nparts):
    lines = source.readlines()
    nline = len(lines)
    for i in range(nparts):
        hit = sample_table(nline, nparts)
        yield part_wrt_hit_table(lines, hit)


class DividedLineSentence(object):
    def __init__(self, source, strategy, npart):
        self.strategy = strategy
        self.npart = npart
        self.source_doc = open(source, 'r')
        if strategy=='partitioning':
            self.all_parts = divide_partitioning(self.source_doc, self.npart)
        elif strategy=='sampling':
            self.all_parts = divide_sampling(self.source_doc, self.npart)

    def __iter__(self):
        self.source_doc.seek(0)
        for part in self.all_parts:
            yield part
            
    def __del__(self):
        self.source_doc.close()
   
