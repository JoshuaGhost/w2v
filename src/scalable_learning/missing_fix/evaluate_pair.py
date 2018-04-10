from copy import deepcopy

from numpy import linalg

from scalable_learning.extrinsic_evaluation.web import evaluate_on_all
from scalable_learning.extrinsic_evaluation.web.analogy import *
from scalable_learning.extrinsic_evaluation.web.datasets.categorization import fetch_AP, fetch_battig, fetch_BLESS
from scalable_learning.extrinsic_evaluation.web.datasets.similarity import fetch_MEN, fetch_WS353, fetch_SimLex999, \
    fetch_MTurk, fetch_RG65, fetch_RW
from scalable_learning.extrinsic_evaluation.web.evaluate import evaluate_analogy, evaluate_categorization, \
    evaluate_similarity, evaluate_on_semeval_2012_2
from scalable_learning.lovv.utils import lovv2web

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
    vocab_test = vocab_intersection[len_train:len_train + len_test]
    if pct_train + pct_test == 1:  # for extrinsic evaluation, the 'evaluation' part is for real missing word
        # interpolation
        vocab0_validation = list(v0.difference(v1))
        vocab1_validation = list(v1.difference(v0))
    else:  # for intrinsic evaluation, the 'evaluation' part consists of some word-vectors that
        # are masked-out, therefore the percent of train data and test data are NOT summed
        # up not to 1
        vocab0_validation = vocab_intersection[len_train + len_test:]
        vocab1_validation = vocab0_validation[-len(vocab0_validation) // 2:]
        vocab0_validation = vocab0_validation[:len(vocab0_validation) // 2]

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


def missing_fix_2_subs(lovvs_train, interpolate, lovvs_test=None, lovvs_validation=None):
    webs_prediction = interpolate(lovvs_train, lovvs_test, lovvs_validation)
    try:
        assert len(webs_prediction[0].vectors) == len(webs_prediction[1].vectors)
    except AssertionError:
        logger.error("Error: size of m1 and m2 are not equal, perhaps \
                some problems occur in the interpolation approach \
                {}".format(interpolate.__name__))
        return -1
    return webs_prediction


def common_words(source, target):
    ps = 0
    pt = 0
    while ps < len(source[0]) and pt < len(target[0]):
        if source[0][ps] < target[0][pt]:
            ps += 1
        elif source[0][ps] > target[0][pt]:
            pt += 1
        else:
            yield (source[0][ps], source[1][ps], target[1][pt])
            ps += 1
            pt += 1


def intrinsic_cv_split(lovvs, pct_train=0.6, pct_test=0.4, sorted_vocab=False):
    source = lovvs[0]
    target = lovvs[1]
    vocab = []
    vs = []
    vt = []
    if sorted_vocab:
        for word, vecs, vect in common_words(source, target):
            vocab.append(word)
            vs.append(vecs)
            vt.append(vect)
    vocab_length = len(vocab)
    test_begin = int(np.round(vocab_length * pct_train))
    if pct_train + pct_test == 1.0:
        validation_begin = vocab_length
    else:
        validation_begin = test_begin + int(vocab_length * pct_test)

    vocab_train = vocab[:test_begin]
    vocab_test = vocab[test_begin:validation_begin]
    vocab_validation = vocab[validation_begin:]

    source_vecs_train = vs[:test_begin]
    source_vecs_test = vs[test_begin:validation_begin]
    source_vecs_validation = vs[validation_begin:]

    target_vecs_train = vt[:test_begin]
    target_vecs_test = vt[test_begin:validation_begin]
    target_vecs_validation = vs[validation_begin:]

    lovvs_train = ((vocab_train, source_vecs_train), (vocab_train, target_vecs_train))
    lovvs_test = ((vocab_test, source_vecs_test), (vocab_test, target_vecs_test))
    lovvs_validation = ((vocab_validation, source_vecs_validation), (vocab_validation, target_vecs_validation))
    return lovvs_train, lovvs_test, lovvs_validation


def common_words_index(part, full):
    pp = 0
    pf = 0
    while pp<len(part) and pf<len(full):
        if part[pp]<full[pf]:
            pp += 1
        elif part[pp]>full[pf]:
            pf += 1
        else:
            yield(pf)
            pp += 1
            pf += 1


def make_index_table(part, vocab):
    index_tabel = np.asarray([0 for w in part])
    for part_idx, vocab_idx in enumerate(common_words_index(part, vocab)):
        index_tabel[part_idx] = vocab_idx
    return index_tabel


def extrinsic_cv_split(lovvs, pct_train=1., pct_test=0., sorted_vocab=False):
    source, target = lovvs

    vocab_train = set(source[0]).intersection(set(target[0]))
    vocab_validation = set(source[0]).difference(vocab_train)
    vocab_train = sorted(list(vocab_train))
    vocab_validation = sorted(list(vocab_validation))
    train_length = len(vocab_train)

    train_index_source = make_index_table(vocab_train, source[0])
    train_index_target = make_index_table(vocab_train, target[0])
    validation_index_source = make_index_table(vocab_validation, source[0])

    source_train = source[1][train_index_source]
    target_train = target[1][train_index_target]
    source_validation = source[1][validation_index_source]

    if pct_train == 1.:
        test_begin = train_length
    else:
        test_begin = int(np.round(train_length * pct_train))

    vocab_train[:test_begin]
    vocab_test = deepcopy(vocab_train[test_begin:])

    source_train = source_train[:test_begin]
    source_test = deepcopy(source_train[test_begin:])
    target_train = target_train[:test_begin]
    target_test = deepcopy(target_train[test_begin:])

    lovvs_train = ((vocab_train, source_train), (vocab_train, target_train))
    lovvs_test = ((vocab_test, source_test), (vocab_test, target_test))
    lovvs_validation = ((vocab_validation, source_validation), (vocab_validation, np.zeros_like(source_validation)))

    return lovvs_train, lovvs_test, lovvs_validation


def eval_intrinsic(lovvs, pct_train=0.6, pct_test=0.4, sorted_vocab=False,
                   interpolate=None, merge_method=None, dataset=None):
    lovvs_origin = deepcopy(lovvs)
    lovvs_train, lovvs_test, lovvs_validation = intrinsic_cv_split(lovvs=lovvs, pct_train=pct_train,
                                                                   pct_test=pct_test, sorted_vocab=sorted_vocab)
    if pct_train + pct_test == 1.0:
        lovvs_predict = interpolate(lovvs_train[0][1], lovvs_test[0][1], lovvs_train[1][1])
    else:
        pass  # neural approaches
    err = 0.
    for origin, prediction in zip(lovvs_origin, lovvs_predict):
        diff = np.asarray(origin[1] - prediction[1])
        err += linalg.norm(diff, 'fro')
    return err


def interpolate_combine(lovvs, interpolate_method, merge, pct_train=0.6, pct_test=0.4, sorted_vocab=False):
    lovv_train, lovv_test, lovv_validation = prep_cross_validation(lovvs=lovvs, pct_train=pct_train, pct_test=pct_test,
                                                                   sorted_vocab=sorted_vocab)
    lovv_predict = missing_fix_2_subs(lovv_train=lovv_train, lovv_test=lovv_test,
                                      lovv_validation=lovv_validation, interpolate=interpolate_method)
    lovv_combined = merge(lovv_predict)
    return lovv_combined


def eval_extrinsic(embeddings, merge, dataset=None):
    lovv_combined = merge(embeddings)
    web_combined = lovv2web(lovv_combined)
    result = evaluate_on_all(web_combined, cosine_similarity=False)
    return result


def eval_demand(embeddings, merge, dataset):
    web_combined = merge(embeddings)

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
        # "ESSLI_2c": fetch_ESSLI_2c(),
        # "ESSLI_2b": fetch_ESSLI_2b(),
        # "ESSLI_1a": fetch_ESSLI_1a()
    }

    if dataset in {'MEN', 'RW', 'WS353', 'WS353S', 'WS353R'}:
        logger.info("Calculating similarity benchmarks")
        data = similarity_tasks[dataset]
        result = evaluate_similarity(web_combined, data.X, data.y, cosine_similarity=False)
        logger.info("Spearman correlation of scores on {} {}".format(dataset, result))
    elif dataset in {'Google'}:
        logger.info("Calculating analogy benchmarks")
        data = analogy_tasks[dataset]
        result = evaluate_analogy(web_combined, data.X, data.y)
        logger.info("Analogy prediction accuracy on {} {}".format(dataset, result))
    elif dataset == 'SemEval2012':
        logger.info("Calculating analogy benchmarks")
        result = evaluate_on_semeval_2012_2(web_combined)['all']
        logger.info("Analogy prediction accuracy on {} {}".format("SemEval2012", result))
    else:
        logger.info("Calculating categorization benchmarks")
        data = categorization_tasks[dataset]
        result = evaluate_categorization(web_combined, data.X, data.y)
        logger.info("Cluster purity on {} {}".format(dataset, result))

    return result
