from scalable_learning.extrinsic_evaluation.web import evaluate_on_all
from scalable_learning.extrinsic_evaluation.web.embedding import Embedding
from copy import deepcopy
import numpy as np


def prep_cross_validation(web1, web2, pct_train=0.6, pct_test=0.4):
    v1, v2 = set(web1.vocabulary), set(web2.vocabulary)
    vocab_intersection = list(v1.intersection(v2))
    len_train = len(vocab_intersection)*pct_train
    len_test = len(vocab_intersection)*pct_test
    vocab_train = vocab_intersection[:len_train]
    vocab_test = vocab_intersection[len_train:len_train+len_test]
    if pct_train + pct_test == 1:
        vocab1_validation = [word for word in v1 if word not in vocab_intersection]
        vocab2_validation = [word for word in v2 if word not in vocab_intersection]
    else:
        vocab1_validation = vocab_intersection[len_train+len_test:]
        vocab2_validation = vocab1_validation

    web1_train, web2_train = deepcopy(web1), deepcopy(web2)
    web1_train.filter(vocab_train)
    web2_train.filter(vocab_train)
    webs_train = (web1_train, web2_train)

    web1_test, web2_test = deepcopy(web1), deepcopy(web2)
    web1_test.filter(vocab_test)
    web2_test.filter(vocab_test)
    webs_test = (web1_test, web2_test)

    web1_validation, web2_validation = deepcopy(web1), deepcopy(web2)
    web1_validation.filter(vocab1_validation)
    web2_validation.filter(vocab2_validation)
    webs_validation = (web1_validation, web2_validation)

    return webs_train, webs_test, webs_validation


def missing_fix_2_subs(webs_train, interpolation_method, webs_test=None, webs_validation=None):
    m1, m2 = interpolation_method(webs_train, webs_test, webs_validation)
    try:
        assert len(m1) == len(m2)
    except AssertionError:
        print ('Error: size of m1 and m2 are not equal, perhapse \
                some problems occur in the interpolation approach \
                {}'.format(interpolation_method.__name__))
        return -1
    return m1, m2


def eval_intrinsic(webs, interpolate_method, merge_method=None, pct_train=0.6, pct_test=0.2):
    web0, web1 = webs
    webs_train, webs_test, webs_validation = prep_cross_validation(web0, web1, pct_train, pct_test)
    webs_predict = missing_fix_2_subs(webs_train, interpolate_method, webs_test, webs_validation)

    diff = np.array(webs_predict[1].select(webs_validation[0].vocabulary)) - np.array(webs_validation[0].vectors)
    diff = diff.dot(diff.T)
    dist = diff.sum(axis=-1)
    err1 = np.sqrt(dist).sum()

    diff = np.array(webs_predict[0].select(webs_validation[1].vocabulary)) - np.array(webs_validation[1].vectors)
    diff = diff.dot(diff.T)
    dist = diff.sum(axis=-1)
    return err1 + np.sqrt(dist).sum()


def eval_extrinsic(ms, interpolation_method, merge_method):
    m1, m2 = missing_fix_2_subs(ms, interpolation_method)
    m_combined = merge_method((m1, m2))
    result = evaluate_on_all(m_combined, cosine_similarity=False)
    return result
