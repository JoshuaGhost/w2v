from scalable_learning.extrinsic_evaluation.web import evaluate_on_all
from scalable_learning.extrinsic_evaluation.web.embedding import Embedding
from copy import deepcopy
import numpy as np


def prep_cross_validation(webs, pct_train=0.6, pct_test=0.4):
    web0, web1 = webs
    v1, v2 = set(web0.vocabulary), set(web1.vocabulary)
    vocab_intersection = list(v1.intersection(v2))
    len_train = int(len(vocab_intersection) * pct_train)
    len_test = int(len(vocab_intersection) * (pct_train + pct_test)) - len_train
    vocab_train = vocab_intersection[:len_train]
    vocab_test = vocab_intersection[len_train:len_train+len_test]
    if pct_train + pct_test == 1:   # for extrinsic evaluation, the 'evaluation' part is for real missing word
                                    # interpolation
        vocab1_validation = [word for word in v1 if word not in vocab_intersection]
        vocab2_validation = [word for word in v2 if word not in vocab_intersection]
    else:                           # for intrinsic evaluation, the 'evaluation' part consists of some word-vectors that
                                    # are masked-out, therefore the percent of train data and test data are NOT summed
                                    # up not to 1
        vocab1_validation = vocab_intersection[len_train+len_test:]
        vocab2_validation = vocab1_validation[-len(vocab1_validation)//2:]
        vocab1_validation = vocab1_validation[:len(vocab1_validation)//2]

    web0_train, web1_train = deepcopy(web0), deepcopy(web1)
    web0_train.filter(vocab_train)
    web1_train.filter(vocab_train)
    webs_train = (web0_train, web1_train)

    web0_test, web1_test = deepcopy(web0), deepcopy(web1)
    web0_test.filter(vocab_test)
    web1_test.filter(vocab_test)
    webs_test = (web0_test, web1_test)

    web0_validation, web1_validation = deepcopy(web0), deepcopy(web1)
    web0_validation.filter(vocab1_validation)
    web1_validation.filter(vocab2_validation)
    webs_validation = (web0_validation, web1_validation)

    return webs_train, webs_test, webs_validation


def missing_fix_2_subs(webs_train, interpolation_method, webs_test=None, webs_validation=None):
    webs_prediction = interpolation_method(webs_train, webs_test, webs_validation)
    try:
        assert len(webs_prediction[0].vectors) == len(webs_prediction[1].vectors)
    except AssertionError:
        print('Error: size of m1 and m2 are not equal, perhapse \
                some problems occur in the interpolation approach \
                {}'.format(interpolation_method.__name__))
        return -1
    return webs_prediction


def eval_intrinsic(webs, interpolate_method, merge_method=None, pct_train=0.6, pct_test=0.2):
    webs_train, webs_test, webs_validation = prep_cross_validation(webs, pct_train, pct_test)
    webs_predict = missing_fix_2_subs(webs_train, interpolate_method, webs_test, webs_validation)

    diff = np.asarray([webs_predict[0][w] - webs[0][w] for w in webs_validation[1].vocabulary])
    diff = diff.dot(diff.T)
    dist = diff.sum(axis=-1)
    err1 = np.sqrt(dist).sum()

    diff = np.asarray([webs_predict[1][w] - webs[1][w] for w in webs_validation[0].vocabulary])
    diff = diff.dot(diff.T)
    dist = diff.sum(axis=-1)
    return err1 + np.sqrt(dist).sum()  # bad evaluation metric for CCA based method. Because CCA transforms both.


def eval_extrinsic(webs, interpolation_method, merge_method, pct_train=0.8, pct_test=0.2):
    webs_train, webs_test, webs_validation = prep_cross_validation(webs, pct_train, pct_test)
    webs_predict = missing_fix_2_subs(webs_train, interpolation_method, webs_test, webs_validation)
    m_combined = merge_method(webs_predict)
    result = evaluate_on_all(m_combined, cosine_similarity=False)
    return result
