#!./env/bin/python
#mapper_dist_res.py

import sys, random
from heapq import heappush, heapreplace

num_samples = int(sys.argv[1])
len_samples = int(sys.argv[2])
H = [[] for i in range(num_samples)]
for line in sys.stdin:
    for sample_idx in range(num_samples):
        r = random.random()
        if len(H[sample_idx]) < len_samples:
            heappush(H[sample_idx], (r, line))
        elif r > H[sample_idx][0][0]:
            heapreplace(H[sample_idx], (r, line))
print H
for sample_idx, sample in enumerate(H):
    for r, line in sample:
        print '{}.{}\t{}'.format(sample_idx, -r, line)

