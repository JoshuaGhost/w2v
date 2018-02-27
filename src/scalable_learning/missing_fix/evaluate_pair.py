from scalable_learning.extrinsic_evaluation.web import evaluate_on_all
import numpy as np


def missing_fix_2_subs(ms, interpolation_method):
    m1, m2 = interpolation_method(ms[0], ms[1])
    try:
        assert len(m1) == len(m2)
    except AssertionError:
        print ('Error: size of m1 and m2 are not equal, perhapse \
                some problems occur in the inerpolation approach \
                {}'.format(interpolation_method.__name__))
        return -1
    return m1, m2


def eval_intrinsic(webs, interpolate_method, merge_method=None, intersection_only=False):
    web1, web2 = webs
    if intersection_only:
        v1 = set(web1.vocabulary)
        v2 = set(web2.vocabulary)
        vocab = list(v1.intersection(v2))
    else:
        vocab = [word for word in web1.vocabulary]
    web1, web2 = missing_fix_2_subs(webs, interpolate_method)
    diff = np.array([web1[word] for word in vocab]) - np.array([web2[word] for word in vocab])
    diff = diff.dot(diff.T)
    dist = diff.sum(axis=-1)
    return np.sqrt(dist).sum()


def eval_extrinsic(ms, interpolation_method, merge_method, intersection_only=False):
    m1, m2 = missing_fix_2_subs(ms, interpolation_method)
    m_combined = merge_method((m1, m2))
    result = evaluate_on_all(m_combined, cosine_similarity=False)
    return result
