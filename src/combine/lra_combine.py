import logging
import numpy as np
import pickle
import os
import sys

from lra import low_rank_align
from sampling import reservoir
from numpy import array

program = os.path.basename(sys.argv[0])
logger = logging.getLogger(program)
logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
logging.root.setLevel(level=logging.INFO)

fvocab = './vocab_bench.txt'

class Selector:
    def __init__(self, n, k, init=None):
        self.n = n
        self.k = k
        if init is not None:
            self.output = init
        else:
            self.output = [i for i in range(k)]
        self.output[-1]-=1
        assert len(self.output) == k

    def find_next(self, p):
        if p == self.k:
            return -1
        if self.output[p] == self.n-(self.k-p):
            return -1
        ret = self.find_next(p+1)
        if ret == -1:
            self.output[p:] = [self.output[p]+i+1 for i, c in enumerate(self.output[p:])]
            return 1
        return ret
    
    def __iter__(self):
        return self

    def next(self):
        ret = self.find_next(0)
        if ret == -1:
            raise StopIteration
        return self.output

if __name__=='__main__':
    debug = True

    n_subs_total = 100
    n_subs_step = 10
    n_subs_min = 10
    dim_sub = 50
    dim_merge = 50

    n_comb = 10

    work_folder = Path('../../models/sampling_100/').resolve()
    subs_folder = (work_folder/Path(subs)).resolve()
    vocab = set()
    word2idx = {}
    vecs = []

    vocab_bench = set(open(fvocab).read().split())
    for part_name in subs_folder.glob('part-*'):
        for line in open(str(part_name)):
            word, vec = line.split(':')
            word = eval(word)
            vec = eval(vec)
            if word in vocab_bench:
                e[word] = len(vecs)
                vecs.append(vec)
                vocab = vocab.union({word})
    vecs = array(vecs)

    for n_subs in range(n_subs_min, n_subs_total, n_subs_step):
        for i in range(n_comb):
            idx_list = reservoir([for i in range(n_subs_total)], n_subs)
            vecs_merge = vecs[:,:dim_sub]
            for nmerged, sub_idx in eumerate(idx_list[1:]):
                Fx, Fy = lra(vecs_merge, vecs[:, sub_idx*dim_sub:(sub_idx+1)*dim_sub], np.eye(vecs.shape[0]))
                vecs_merge = (Fx*(nmerged + 1)+Fy)/(nmerged + 2)



