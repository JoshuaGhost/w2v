#!./env/bin/python

import sys
import pandas as pd

results_pca = pd.DataFrame()
results_concat = pd.DataFrame()

for line in sys.stdin:
    nsubs, results = line.split('\t', 1)
    results = eval(results)
    results_concat = results_concat.append(pd.DataFrame.from_dict(results['concat']))
    results_pca = results_pca.append(pd.DataFrame.from_dict(results['pca']))

print repr({'concat':results_concat.to_dict(), 'pca':results_pca.to_dict()})
    
