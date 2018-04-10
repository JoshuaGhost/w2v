#!/usr/bin/python

import argparse
import functools
import sys

import gensim.models.word2vec

from scalable_learning.corpus_divider import divide_corpus
from scalable_learning.extrinsic_evaluation.web.evaluate import evaluate_on_all
from scalable_learning.merge import pca, concat, lra
from scalable_learning.missing_fix.evaluate_pair import eval_intrinsic, eval_extrinsic, eval_demand, extrinsic_cv_split
from scalable_learning.missing_fix.interpolate import global_transform, fill_zero, orthogonal_procrustes, \
    cca, orthogonal_procrustes_regression, ordinary_least_square
from scalable_learning.utils import read_wv_csv
from utils import *
from config import SUB_MIN_COUNT, DIM, NUM_WORKERS, ERR_THRESHOLD

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
logging.root.setLevel(level=logging.INFO)
logger.info("running %s" % ' '.join(sys.argv))


def eval_interpolate(lovvs, interpolate, evaluate, merging_method, benchmark=None, sorted_vocab=False):
    try:
        assert len(lovvs) == 2
    except AssertionError:
        print("not implemented!")
        return -1
    logger.info('begin to run {} evaluation for interpolation method {}, the merging method is {}'.format(
        evaluate.__name__, interpolate.__name__, merging_method.__name__))
    lovvs_train, lovvs_test, lovvs_validation = extrinsic_cv_split(lovvs)
    source_predict, target_predict = interpolate(source_train=lovvs_train[0][1],
                                                 source_test =lovvs_validation[0][1],
                                                 target_train=lovvs_train[1][1])
    vocabulary = lovvs_train[0][0] + lovvs_validation[0][0]
    source_web = Embedding(vocabulary=OrderedVocabulary(vocabulary), vectors=source_predict)
    target_web = Embedding(vocabulary=OrderedVocabulary(vocabulary), vectors=target_predict)
    ret = evaluate((source_web, target_web), merge=merging_method, dataset=benchmark)
    return ret


def make_index_table(part, vocab):
    index_tabel = np.asarray([0 for w in part])
    for part_idx, vocab_idx in enumerate(common_words_index(part, vocab)):
        index_tabel[part_idx] = vocab_idx
    return index_tabel


if __name__ == '__main__':
    '''
    python main.py operation num_subs 
        sub_folder sub_filename sub_extension 
        interpolation_method evaluation_method merge_method 
        log_file_name
    '''
    debug = False
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', help='type of task (evalfix2/eval/bench/fix2/div/fixall)')
    parser.add_argument('-l', help='logging file', default='./log.txt')
    parser.add_argument('-c', help='count of the sub models', type=int, default=2)
    parser.add_argument('-d', help='input directory')
    parser.add_argument('--debug', help='debug switch', dest='debug', action='store_true')
    parser.add_argument('-f', help='(prefix of) filenames')
    parser.add_argument('-T', help='type of inputfile')
    parser.add_argument('-e', help='extention of the filenames', default='')
    parser.add_argument('--imethod', help='interpolation method a/c/p/z', default='z')
    parser.add_argument('--emethod', help='evaluation method', default='i')
    parser.add_argument('--mmethod', help='merging method', default='c')
    parser.add_argument('--cfolder', help='the folder where the sub-corpora file is saved', default='')
    parser.add_argument('--cname_source', help='filename of source sub-corpus', default='')
    parser.add_argument('--dump_folder', help='directory where all the trained models are dumped', default='/tmp/zzj/')
    parser.add_argument('--cname_target', help='filename of target sub-corpus', default='')
    parser.add_argument('--vbfolder', help='folder where vocabularies of benchmarks are saved', default='')
    parser.add_argument('--vbname', help='filename of benchmark vocabulary', default='')
    parser.add_argument('--vocab_union_file', help='dumping file for vocabularies in union', default='vocab_union')
    args = parser.parse_args()

    if args.debug:
        DIM = 5
        SUB_MIN_COUNT = 1

    task, log_fname = args.t, args.l
    tasks = {'evalfix2': 'evaluate_2_interpolation',
             'eval': 'evaluate_single_sub',
             'bench': 'mask_benchmark',
             'fix2': 'interpolate',
             'div': 'divide',
             'fixall': 'interpolate_all'}
    task = tasks[task]

    if task == "evaluate_2_interpolation":
        count_subs = args.c
        subs_folder, filename, extension, input_type = args.d, args.f, args.e, args.T
        imethod, emethod, mmethod = args.imethod, args.emethod, args.mmethod
        logger.info("loading {} embeddings from {}".format(count_subs, subs_folder))
        subs = load_embeddings(subs_folder, filename=filename, extension=extension, use_norm=True,
                               input_format=input_type)
        result = eval_interpolate(webs=subs, interpolate=imethod, evaluate=emethod, merging_method=mmethod)
        result['num_sub'] = 2
        with open('results-{}.csv'.format(imethod), 'a+') as fout:
            fout.write(result.to_csv())

    elif task == "evaluate_single_sub":
        model_folder, filename, input_type = args.d, args.f, args.T
        fname = model_folder + '/' + filename
        model = read_wv_csv(fname)
        result = evaluate_on_all(model, cosine_similarity=False)
        result['num_sub'] = 1
        with open('result.csv', 'a+') as fout:
            fout.write(result.to_csv())

    elif task == 'mask_benchmark':
        dump_folder = args.dump_folder
        if not os.path.isdir(dump_folder):
            os.makedirs(dump_folder)
        source_dump = args.dump_folder + '/' + 'source-dim50.csv'
        cname_source = args.cfolder + '/' + args.cname_source
        model_source = load_or_train(fname_corpus=cname_source, size=DIM, negative=N_NS, workers=NUM_WORKERS,
                                     min_count=SUB_MIN_COUNT, use_norm=True, dump_file=source_dump)
        dataset = args.vbname
        vbname = args.vbfolder + '/' + dataset
        logger.info('loading words from dataset {}'.format(dataset))
        with codecs.open(vbname, 'r', encoding='utf8', buffering=1) as fvb:
            vocab_benchmark = set(word.strip() for word in fvb)

        cname_target = args.cfolder + '/' + args.cname_target

        target0_dump = dump_folder + '/' + 'target0-{}-dim50.csv'.format(dataset)
        model_target0 = load_or_train(fname_corpus=cname_target, filter_vocab=vocab_benchmark, size=DIM, negative=N_NS,
                                      workers=NUM_WORKERS, min_count=SUB_MIN_COUNT, use_norm=True,
                                      dump_file=target0_dump)

        vocab_target = gensim.models.word2vec.Word2VecVocab(min_count=SUB_MIN_COUNT, null_word=1)
        vocab_target.scan_vocab(LineSentence(cname_target))
        vocab_target_sorted = list(vocab_target.raw_vocab.keys())
        vocab_benchmark = vocab_benchmark.intersection(set(vocab_target_sorted))
        vocab_target_sorted.sort(key=(lambda x: vocab_target.raw_vocab[x]), reverse=False)
        size_benchmark = len(vocab_benchmark)
        for word in vocab_target_sorted:
            if word in vocab_benchmark:
                vocab_benchmark.remove(word)
            if len(vocab_benchmark) <= size_benchmark // 2:
                break

        target1_dump = dump_folder + '/' + 'target1-{}-dim50.csv'.format(dataset)
        model_target1 = load_or_train(fname_corpus=cname_target, filter_vocab=vocab_benchmark, size=DIM, negative=N_NS,
                                      workers=NUM_WORKERS, min_count=SUB_MIN_COUNT, use_norm=True,
                                      dump_file=target1_dump)

        models_target = [model_target0, model_target1]
        with open('source-target.csv', 'a+') as fout:
            head = [em + im + t for t in '12' for em in 'ie' for im in 'acpz'] + \
                   ['source', 'target0', 'target1', 'dataset']
            fout.write(','.join(head) + '\n')
            eval_methods = {'i': eval_intrinsic,
                            'e': eval_extrinsic,
                            'd': eval_demand}
            interpolation_methods = {'g': global_transform,
                                     'c': cca,
                                     'p': orthogonal_procrustes,
                                     'z': fill_zero}
            merging_methods = {'p': pca,
                               'c': concat,
                               'l': lra}
            eval_names = {'i': 'intrinsic',
                          'd': 'dataset-shadowed extrinsic'}
            interpolation_names = {'g': 'global transformation',
                                   'c': 'CCA',
                                   'p': 'orthogonal Procrustes',
                                   'z': 'trivial filling zeros'}
            for model_idx in (0, 1):
                for em in 'd':
                    for im in 'gcpz':
                        result = eval_interpolate((model_source, models_target[model_idx]),
                                                  interpolate=interpolation_methods[im],
                                                  merging_method=merging_methods['c'],
                                                  evaluate=eval_methods[em],
                                                  benchmark=dataset,
                                                  sorted_vocab=True)
                        logger.info('result of {} evaluation using approach {} is: {}'.format(eval_names[em],
                                                                                              interpolation_names[im],
                                                                                              result))
                        fout.write('{}, '.format(result))
            fout.write('{}\n'.format(dataset))
        result = 'written in source-target.csv'
    elif task == 'interpolate_all':
        subs_folder, filename, extension, input_type = args.d, args.f, args.e, args.T
        imethod, emethod, mmethod = args.imethod, args.emethod, args.mmethod
        lovvs = load_embeddings(folder=subs_folder, filename=filename, extension=extension, use_norm=True,
                                input_format=input_type, output_format='lovv')
        dump_folder = args.dump_folder
        vocab_union_file = dump_folder + '/' + args.vocab_union_file
        vocab_union = []
        if os.path.isfile(vocab_union_file):
            with codecs.open(vocab_union_file, 'r', encoding='utf-8') as fdump:
                vocab_union = [word.strip() for word in fdump]
        else:
            vocab_sets = map(set, [vv[1] for vv in lovvs])
            vocab_intersection = functools.reduce(lambda a, b: a.intersection(b), vocab_sets)
            vocab_union = functools.reduce(lambda a, b: a.union(b), vocab_sets)
            vocab_union = sorted(list(vocab_union))
            with codecs.open(vocab_union_file, 'w+', encoding='utf-8') as fdump:
                for word in vocab_union:
                    fdump.write(word+'\n')
        for lovv in lovvs:
            lovv.append(make_index_table(lovv[0], vocab_union))
        vocab_count = functools.reduce(lambda a, b: a + b, [lovv[2].astype(int) for lovv in lovvs])
        vocab_count = np.asarray(vocab_count).reshape(len(vocab_union, 1))
        regress = ordinary_least_square if args.r == 'sols' else orthogonal_procrustes_regression
        err_total = 1000000.
        dim_sub = lovvs[0][1].shape[-1]
        y = np.random.random((len(vocab_union), dim_sub))
        predict = [np.zeros((dim_sub, dim_sub)) for i in lovvs]
        while err_total > ERR_THRESHOLD:
            err_total = 0
            for idx, lovv in enumerate(lovvs):
                predict[idx], err = regress(lovv[1], y[lovv[2]])
                err_total += err
            y = np.zeros_like(y)
            for idx, _ in enumerate(lovvs):
                y[lovv[2]] += predict[idx]
            y = y / vocab_count
            y = y / np.linalg.norm(y, axis=0)
            err_total = err_total / np.sqrt(len(vocab_union) * dim_sub)
        result = err_total
    else:
        tasks = {'divide': divide_corpus}
        result = tasks[task](sys.argv[2:-1])

    with open(log_fname, 'a+') as fout:
        fout.write('configuration:\n{}\nresult:{}\n'.format(sys.argv, result))
