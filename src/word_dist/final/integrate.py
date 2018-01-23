files = ['partitionsize-df_bigram.csv', 'partitionsize-df_word.csv', 'samp-kld_bigram.csv', 'samp-kld_word.csv']
import pandas as pd
results = pd.DataFrame(columns='sample_size partition_size KL_Divergence'.split())
for fname in files:
    r = pd.DataFrame.from_csv(fname)
    if hasattr(r, 'sample_size'):
        r['sample_size']=[float(e.split('/')[0])/float(e.split('/')[1]) for e in r['sample_size']]
        r = r.groupby(['sample_size']).mean()
    else:
        r['partition_size'] = [float(e.split('/')[0])/float(e.split('/')[1]) if hasattr(e, 'split') else e for e in r['partition_size']]
        r = r.groupby(['partition_size']).mean()
    r.to_csv('pandas_'+fname)
    print r
    results = results.append(r)
results.to_csv('all.csv')
