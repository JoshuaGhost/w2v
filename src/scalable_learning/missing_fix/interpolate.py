import numpy as np
from scalable_learning.extrinsic_evaluation.web.vocabulary import OrderedVocabulary


def linear_transform(m1, m2):
    pass


def fill_zero(web1, web2):
    v1, v2 = set(web1.vocabulary), set(web2.vocabulary)
    v_union = list(v1.union(v2))
    vocab = OrderedVocabulary(v_union)
    dim1, dim2 = web1.vectors.shape[-1], web2.vectors.shape[-1]
    vecs1, vecs2 = [], []
    for word in v_union:
        if word not in web1.vocabulary:
            vecs1.append(np.zeros((1, dim1)))
        else:
            vecs1.append(web1.vectors[web1.vocabulary[word]])
        if word not in web2.vocabulary:
            vecs2.append(np.zeros((1, dim2)))
        else:
            vecs2.append(web2.vectors[web2.vocabulary[word]])
    web1.vocabulary, web2.vocabulary = vocab, vocab
    web1.vectors, web2.vectors = np.vstack(vecs1), np.vstack(vecs2)
    return web1, web2


def tcca(ms):
    pass


def gcca(ms):
    pass