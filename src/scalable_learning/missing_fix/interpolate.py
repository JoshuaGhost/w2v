import numpy as np


def linear_transform(m1, m2):
    pass


def fill_zero(m1, m2):
    v1 = set(m1.keys())
    v2 = set(m2.keys())
    v_union = list(v1.union(v2))
    dim1 = m1[list(v1)[0]].shape[-1]
    dim2 = m2[list(v2)[0]].shape[-1]
    for word in v_union:
        if word not in v1:
            m1[word] = np.zeros((1, dim1))
        if word not in v2:
            m2[word] = np.zeros((1, dim2))
    return m1, m2


def tcca(ms):
    pass


def gcca(ms):
    pass