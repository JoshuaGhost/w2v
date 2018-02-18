#!./env/bin/python
#reducer_dist_res.py

import sys

k = int(sys.argv[1])
c = 0

for line in sys.stdin:
    l = line.strip()
    if len(l) == 0: #debug in localhost, sorting produces multiple empty lines
        continue
    (key, val) = line.split('\t', 1)
    print val,
    c += 1
    if c == k:
        break
for line in sys.stdin:
    pass
