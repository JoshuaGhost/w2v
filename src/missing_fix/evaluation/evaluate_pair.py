from random import sample
from ..interpolate import linear_transform
from collections import OrderedDict
from ..evaluate.web import evaluate_on_all

def missing_fix(ms, interpolate):
    m1, m2 = interpolate(ms[0], ms[1])
    try:
        assert len(m1) == len(m2)
    except AssertionError:
        print ('Error: size of m1 and m2 are not equal, perhapse \
                some problems occur in the inerpolation approach \
                {}'.format(interpolation.__name__))
        return -1
    return m1, m2

def eval_intrinsic(ms, interpolate):
    m1, m2 = missing_fix(ms, interpolate)
    vocab = [word for word in m1.keys()]
    vecs1, vecs2 = np.array([m1[word] for word in vocab]), np.array([m2[word] for word in vocab])
    diff = vecs1 - vecs2
    diff = diff.dot(diff.T)
    dist = diff.sum(axis=-1)
    return np.sqrt(dist).sum()

def eval_extrinsic(m1, m2, interpolate):
    m1, m2 = missing_fix(ms, interpolate)
    result = evaluate_on_all(combine_models(m1, m2), cosine_similarity=False)
    return result
    
