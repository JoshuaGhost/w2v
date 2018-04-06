import logging
from copy import deepcopy

import numpy as np
from numpy import linalg

from scalable_learning.extrinsic_evaluation.web import evaluate_on_all
from scalable_learning.extrinsic_evaluation.web.analogy import *
from scalable_learning.extrinsic_evaluation.web.datasets.categorization import fetch_AP, fetch_battig, fetch_BLESS
from scalable_learning.extrinsic_evaluation.web.datasets.similarity import fetch_MEN, fetch_WS353, fetch_SimLex999, \
    fetch_MTurk, fetch_RG65, fetch_RW
from scalable_learning.extrinsic_evaluation.web.evaluate import evaluate_analogy, evaluate_categorization, \
    evaluate_similarity, evaluate_on_semeval_2012_2

logger = logging.getLogger(__name__)


def prep_cross_validation(webs, pct_train=0.6, pct_test=0.4):
    logger.info('preparing dataset for cross validation, %train={}, %test={}'.format(pct_train, pct_test))
    web0, web1 = webs
    v0, v1 = set(web0.vocabulary), set(web1.vocabulary)
    vocab_intersection = list(v0.intersection(v1))
    intersection_size = len(vocab_intersection)
    logger.info("size of v0: {}, size of v1: {}, sizeof intersection: {}".format(len(v0), len(v1),
                                                                                 intersection_size))
    len_train = int(intersection_size * pct_train)
    len_test = int(intersection_size * (pct_train + pct_test)) - len_train
    vocab_train = vocab_intersection[:len_train]
    vocab_test = vocab_intersection[len_train:len_train+len_test]
    if pct_train + pct_test == 1:   # for extrinsic evaluation, the 'evaluation' part is for real missing word
                                    # interpolation
        vocab0_validation = list(v0.difference(v1))
        vocab1_validation = list(v1.difference(v0))
    else:                           # for intrinsic evaluation, the 'evaluation' part consists of some word-vectors that
                                    # are masked-out, therefore the percent of train data and test data are NOT summed
                                    # up not to 1
        vocab0_validation = vocab_intersection[len_train+len_test:]
        vocab1_validation = vocab0_validation[-len(vocab0_validation)//2:]
        vocab0_validation = vocab0_validation[:len(vocab0_validation)//2]

    web0_train, web1_train = deepcopy(web0), deepcopy(web1)
    web0_train.filter(vocab_train)
    web1_train.filter(vocab_train)
    webs_train = (web0_train, web1_train)

    web0_test, web1_test = deepcopy(web0), deepcopy(web1)
    web0_test.filter(vocab_test)
    web1_test.filter(vocab_test)
    webs_test = (web0_test, web1_test)

    web0_validation, web1_validation = deepcopy(web0), deepcopy(web1)
    web0_validation.filter(vocab0_validation)
    web1_validation.filter(vocab1_validation)
    webs_validation = (web0_validation, web1_validation)

    return webs_train, webs_test, webs_validation


def missing_fix_2_subs(webs_train, interpolation_method, webs_test=None, webs_validation=None):
    webs_prediction = interpolation_method(webs_train, webs_test, webs_validation)
    try:
        assert len(webs_prediction[0].vectors) == len(webs_prediction[1].vectors)
    except AssertionError:
        logger.error("Error: size of m1 and m2 are not equal, perhaps \
                some problems occur in the interpolation approach \
                {}".format(interpolation_method.__name__))
        return -1
    return webs_prediction


def eval_intrinsic(webs, interpolate_method, merge_method=None, pct_train=0.6, pct_test=0.2):
    webs_train, webs_test, webs_validation = prep_cross_validation(webs, pct_train, pct_test)
    webs_predict = missing_fix_2_subs(webs_train, interpolate_method, webs_test, webs_validation)
    diff = np.asarray([webs_predict[0][w] - webs[0][w] for w in webs_validation[1].vocabulary])
    err = linalg.norm(diff, 'fro')
    diff = np.asarray([webs_predict[1][w] - webs[1][w] for w in webs_validation[0].vocabulary])
    err += linalg.norm(diff, 'fro')
    logger.info("Intrinsic evaluation completed. Frobenius error: {}".format(err))
    return err # bad evaluation metric for CCA based method. Because CCA transforms both.


def interpolate_combine(webs, interpolation_method, merge_method, pct_train=0.8, pct_test=0.2):
    webs_train, webs_test, webs_validation = prep_cross_validation(webs, pct_train, pct_test)
    webs_predict = missing_fix_2_subs(webs_train, interpolation_method, webs_test, webs_validation)
    m_combined = merge_method(webs_predict)
    return m_combined


def eval_extrinsic(m_combined):
    result = evaluate_on_all(m_combined, cosine_similarity=False)
    return result


def eval_demand(m_combined, dataset='MEN'):
    similarity_tasks = {
        "MEN": fetch_MEN(),
        "WS353": fetch_WS353(),
        "WS353R": fetch_WS353(which="relatedness"),
        "WS353S": fetch_WS353(which="similarity"),
        "SimLex999": fetch_SimLex999(),
        "RW": fetch_RW(),
        "RG65": fetch_RG65(),
        "MTurk": fetch_MTurk(),
    }

    analogy_tasks = {
        "Google": fetch_google_analogy(),
        "MSR": fetch_msr_analogy()
    }

    categorization_tasks = {
        "AP": fetch_AP(),
        "BLESS": fetch_BLESS(),
        "Battig": fetch_battig(),
        #"ESSLI_2c": fetch_ESSLI_2c(),
        #"ESSLI_2b": fetch_ESSLI_2b(),
        #"ESSLI_1a": fetch_ESSLI_1a()
    }

    if dataset in {'MEN', 'RW', 'WS353', 'WS353S', 'WS353R'}:
        logger.info("Calculating similarity benchmarks")
        data = similarity_tasks[dataset]
        result = evaluate_similarity(m_combined, data.X, data.y, cosine_similarity=False)
        logger.info("Spearman correlation of scores on {} {}".format(dataset, result))
    elif dataset in {'Google'}:
        logger.info("Calculating analogy benchmarks")
        data = analogy_tasks[dataset]
        result = evaluate_analogy(m_combined, data.X, data.y)
        logger.info("Analogy prediction accuracy on {} {}".format(dataset, result))
    elif dataset == 'SemEval2012':
        logger.info("Calculating analogy benchmarks")
        result = evaluate_on_semeval_2012_2(m_combined)['all']
        logger.info("Analogy prediction accuracy on {} {}".format("SemEval2012", result))
    else:
        logger.info("Calculating categorization benchmarks")
        data = categorization_tasks[dataset]
        result = evaluate_categorization(m_combined, data.X, data.y)
        logger.info("Cluster purity on {} {}".format(dataset, result))

    return result
