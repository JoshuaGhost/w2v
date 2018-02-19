#!./env/bin/python
from numpy import array
import sys

NUM_SPLIT=10

currentword = None

buffer = {}

npart = int(sys.argv[1])

for line in sys.stdin:
    word, vec = line.split('\t')
    mapper_idx, vec = vec.split('#')

    word = word.strip()
    mapper_idx = int(mapper_idx)
    vec = vec.strip()
    '''
        damn hadoop may feed a reducer with
        inputs with different keys
    '''
    if not currentword or currentword != word:
        if currentword and len(buffer) == npart:
            print currentword + ':' + ','.join([buffer[idx] for idx in buffer])
        buffer = {}
        currentword = word
    buffer[mapper_idx] = vec
