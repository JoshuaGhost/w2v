#!/env/bin/python

import sys

dim_sub_model = 500
n_sub_models = 100

if __name__ == "__main__":
    for idx, bundle in enumerate(open(filenames)):
        sub_vocab = []
        vecs = []
        for embedding in open(bundle):
            word, vec = embedding.split(':')
            sub_vocab.append(word)
            vecs.append(vec.split(','))
            vec = vec.split(',')
        for i in range(0, n_sub_models):
            v = []
            f = open('submodel-'+str(i), 'a+')
            for word in sub_vocab[]:
                f.write(word+':'+','.join(vec[i*dim_sub_model: (i+1)*dim_sub_model])+'\n')
            f.close()

