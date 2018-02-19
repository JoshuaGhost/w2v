#!/env/bin/python

import sys

dim_sub_model = 50
n_sub_models = 10
filenames = 'filenames'
submodel_folder = 'submodels/'

if __name__ == "__main__":
    sub_models = [open(submodel_folder+'submodel-'+str(i), 'a+') for i in range(0, n_sub_models)]
    for bundle in open(filenames):
        for embedding in open(bundle.strip()):
            if len(embedding.strip()) == 0:
                continue
            word, vec = embedding.split(':')
            vec = vec.split(',')
            for i, f in enumerate(sub_models):
                f.write(word+':'+','.join(vec[i*dim_sub_model: (i+1)*dim_sub_model])+'\n')
    
    for f in sub_models:
        f.close()

