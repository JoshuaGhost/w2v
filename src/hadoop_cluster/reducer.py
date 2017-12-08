#!./env/bin/python
from numpy import array
import sys

NUM_SPLIT=10

currentword = None

buffer = ['0' for i in range(NUM_SPLIT)]

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
        if currentword and len(buffer)==NUM_SPLIT:
            print currentword + ':' + ','.join(buffer)
        buffer = ['0' for i in range(NUM_SPLIT)]
        currentword = word
    else:
        buffer[mapper_idx] = vec
