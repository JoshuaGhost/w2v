#!./env/bin/python
#mapper_dist_res.py

import sys, random
from heapq import heappush, heapreplace

k = int(sys.argv[1])
H = []
for l in sys.stdin:
    filename = l.strip()
    for line in open(filename):
        r = random.random()
        if len(H) < k:
            heappush(H, (r, line))
        elif r > H[0][0]:
            heapreplace(H, (r, line))
    for (r, x) in H:
        print '{}\t{}'.format(-r, x)
