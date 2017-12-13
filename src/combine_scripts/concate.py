from utils import load_embeddings 
from pickle import dump

#folder_in = '/home/zijian/workspace/master_thesis/w2v/temp/models/sampling/sub/'
folder_in = '/home/zijian/workspace/master_thesis/w2v/temp/models/documentwise/subs/'
filename = 'article.txt.0'
extension = '.w2v'
nmodels = 10
norm = False
mean_corr = False
#folder_out = '/home/zijian/workspace/master_thesis/w2v/temp/models/sampling/combined/'
folder_out = '/home/zijian/workspace/master_thesis/w2v/temp/models/documentwise/combined/'
dump_name = 'concate'

vocab, vecs = load_embeddings(folder_in, filename, extension, nmodels, norm)
d = dict(zip(vocab, vecs))
dump(d, open(folder_out+'/'+dump_name+'.pkl', 'w+'))
