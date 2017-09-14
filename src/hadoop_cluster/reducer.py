#!./env/bin/python
from numpy import array
import sys

vec_sum = array(0)
word_self = ''

for line in sys.stdin:
    word, vec = line.split('\t', 1)
    '''
    vec = array([float(num) for num in vec.split(',')])
    if vec_sum.size == 1:
        vec_sum = vec
        word_self = word
    else:
        vec_sum += vec

print '\t'.join((word_self, ','.join([str(num) for num in vec_sum])))
    '''
    print '\t'.join((word, vec))
