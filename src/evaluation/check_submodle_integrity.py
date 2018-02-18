from utils import load_embeddings

for folder in ['subs/', 'right_subs/']:
    for i in range(6, 100, 18):
        print('loading submodel /tmp/zzj/sampling_100_dim_500/{}submodel-{}'.format(folder, [j for j in range(i, i+18+1)]))
        vocab, vecs = load_embeddings('/tmp/zzj/sampling_100_dim_500/'+folder, 'submodel-', '', False, 'submodels', [j for j in range(i, i+18+1)])
        vocab = None
        vecs = None

