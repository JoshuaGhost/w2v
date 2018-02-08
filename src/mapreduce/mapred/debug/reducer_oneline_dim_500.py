#!./env/bin/python
from numpy import array
import sys
import logging

FORMAT = '%(asctime)-15s %(message)s'
logging.basicConfig(format=FORMAT)
logger = logging.getLogger('reducer')
logger.setLevel(logging.INFO)

currentword = None

vec_buffer = []

npart = int(sys.argv[1])
for line in sys.stdin:
    word, _, vec = line.split('?')
    word = word.strip()
    vec = vec.strip()
    '''
        damn hadoop may feed a reducer with
        inputs with different keys
    '''
    if currentword is None or currentword != word:
        if currentword and len(vec_buffer) == npart:
            print currentword + ':' + ','.join(vec_buffer)
        currentword = word
        vec_buffer = []
    vec_buffer.append(vec)
    logger.info("word: {}, idx: {}, vec[0]: {}".format(word, _, len(vec_buffer)))
else:
    if len(vec_buffer) == npart:
        print currentword + ':' + ','.join(vec_buffer)

