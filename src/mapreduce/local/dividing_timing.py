#!./env/bin/python
from itertools import islice
from math import log
import sys
import copy, random
import os
import time
from pathlib2 import Path

strategy = 'sampling'
nparts = 10
origin_corpus_folder = Path('../').resolve()
origin_corpus_fname = origin_corpus_folder/Path('article_4227933_lines.txt')
work_path = Path('../subcorpora').resolve()
work_path = work_path/Path(strategy+'_'+str(nparts))
fname_time = work_path/Path('time.csv')
sub_corpora_path = work_path/Path('sub_corpora')

nlines = 4227933 
size_in_byte = os.stat(str(origin_corpus_fname)).st_size

lines_per_part = nlines/nparts+1
byte_per_part = size_in_byte/nparts+1

out_name = 'article'
out_ext = 'txt'


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
    source = open(str(origin_corpus_fname), 'r')
    time_sec = []
    for idx in range(nparts):
        timestamp = -time.time()
        fname_out = '.'.join([strategy,out_name, str(idx).zfill(int(log(nparts)/log(10)+1)), out_ext])
        fname_out = sub_corpora_path/Path(fname_out)
        if os.path.isfile(str(fname_out)):
            continue
        fout = open(str(fname_out), 'w+')
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
        timestamp += time.time()
        time_sec.append(timestamp)
        ftime = open(str(fname_time), 'w+')
        for idx, t in enumerate(time_sec):
            ftime.write(str(idx)+', '+str(t)+'\n')
        ftime.close()
    source.close()

