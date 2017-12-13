#!./env/bin/python
from itertools import islice
from math import log
import sys
import copy, random
import os


def reservoir(source, nsamples):
    ret = []
    for idx, sentence in enumerate(source.readlines()):
        if len(ret)<nsamples:
            ret.append(sentence)
        else:
            if int(random.uniform(0, idx)) < nsamples:
                ret[int(random.uniform(0, nsamples))] = sentence
    return ret


if __name__ == '__main__':
    source_name = 'article.txt'
    nparts = 100

    nlines = int(os.popen('wc '+source_name).read().split()[0])
    size_in_byte = os.stat(source_name).st_size
    lines_per_part = nlines/nparts+1
    #lines_per_part = nlines
    byte_per_part = size_in_byte/nparts+1

    out_name = 'article'
    out_ext = 'txt'

    strategy = 'sampling'

    source = open(source_name, 'r')
    for idx in range(nparts):
        fout = open('.'.join([strategy,out_name,str(idx).zfill(int(log(nparts)/log(10)+1)),out_ext]), 'w+')
        if strategy == 'skip':
            source.seek(0)
            sentences = [sentence for sentence in islice(source.readlines(), idx, None, nparts)]
        elif strategy == 'sampling':
            source.seek(0)
            sentences = [sentence for sentence in reservoir(source, lines_per_part)]
        elif strategy == 'chunk':
            sentences = source.read(byte_per_part)
        fout.writelines(sentences)
        fout.close()
    source.close()

