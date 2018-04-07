#!/usr/bin/python

import argparse
import codecs
import logging
import sys

import os.path
from gensim.models.word2vec import Word2Vec, Word2VecVocab, LineSentence
from gensim.utils import RULE_DISCARD, RULE_KEEP

from scalable_learning.corpus_divider import DividedLineSentence
from scalable_learning.extrinsic_evaluation.web.evaluate import evaluate_on_all
from scalable_learning.merge import pca, concat, lra
from scalable_learning.missing_fix.evaluate_pair import interpolate_combine, eval_intrinsic, eval_extrinsic, eval_demand
from scalable_learning.missing_fix.interpolate import affine_transform, fill_zero, orthogonal_procrustes, cca
from scalable_learning.utils import load_embeddings, read_wv_csv, gensim2web, web2csv

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
logging.root.setLevel(level=logging.INFO)
logger.info("running %s" % ' '.join(sys.argv))

DIM = 500
N_NS = 5
MIN_COUNT = 100
NPARTS = 10
SUB_MIN_COUNT = 100 // NPARTS

NUM_WORKERS =18 
def eval_interpolate(webs, im, em, mm, dataset=None):
    count_subs = len(webs)
    eval_methods = {'i': eval_intrinsic, 'e': eval_extrinsic, 'd': eval_demand}
    interpolate_methods = {'a': affine_transform, 'p': orthogonal_procrustes, 'c': cca, 'z': fill_zero}
    merge_methods = {'p': pca, 'c': concat, 'l': lra}
    try:
        assert count_subs == 2
    except AssertionError:
        print("not implemented!")
        return -1
    logger.info('begin to run {} evaluation for interpolation method {}, the merging method is {}'.format(
        eval_methods[em].__name__, interpolate_methods[im], merge_methods[mm]))
    if em == 'i':
        ret = eval_intrinsic(webs[:2], interpolate_methods[im], merge_methods[mm])
    elif em == 'e':
        m_combined = interpolate_combine(webs[:2], interpolate_methods[im], merge_methods[mm])
        ret = eval_extrinsic(m_combined)
    elif em == 'd':
        m_combined = interpolate_combine(webs[:2], interpolate_methods[im], merge_methods[mm])
        ret = eval_demand(m_combined, dataset)
    else:
        ret = -1
    return ret


def interpolate(argvs):
    pass


def divide_corpus(argvs):
    input_fname, strategy, npart, output_folder, output_fname = argvs
    try:
        assert strategy in {'sampling', 'partitioning'}
    except AssertionError:
        logger.error('dividing strategy should be either "sampling" or "partitioning"')
    output_fname = '/'.join((output_folder, strategy, npart, output_fname))
    npart = int(npart)

    logger.info('start to divide corpora')
    logger.info('divide corpus file [{}] into [{}] part(s) with strategy [{}]'.format(input_fname, npart, strategy))
    logger.info('divided sub_corpora saved as [{}]'.format(output_fname))

    lineSentences = DividedLineSentence(input_fname, strategy, npart)
    with open(output_fname, 'w+') as fout:
        for idx, part in enumerate(lineSentences):
            output_lines = [line for line in part]
            fout.write('?'.join(output_lines) + '\n')
            logger.info('sub-corpus No. {} saved'.format(idx))
    return output_fname


if __name__ == '__main__':
    '''
    python main.py operation num_subs 
        sub_folder sub_filename sub_extension 
        interpolation_method evaluation_method merge_method 
        log_file_name
    '''
    debug = False
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', help='type of task (evalfix2/eval/bench/fix2/div)')
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
    args = parser.parse_args()

    if args.debug:
        DIM = 5
        SUB_MIN_COUNT = 1

    task, log_fname = args.t, args.l
    tasks = {'evalfix2': 'evaluate_2_interpolation',
             'eval': 'evaluate_single_sub',
             'bench': 'mask_benchmark',
             'fix2': 'interpolate',
             'div': 'divide'}
    task = tasks[task]

    if task == "evaluate_2_interpolation":
        count_subs = args.c
        subs_folder, filename, extension, input_type = args.d, args.f, args.e, args.T
        imethod, emethod, mmethod = args.imethod, args.emethod, args.mmethod
        logger.info("loading {} embeddings from {}".format(count_subs, subs_folder))
        subs = load_embeddings(subs_folder, filename=filename, extension=extension, norm=True, arch=input_type)
        result = eval_interpolate(webs=subs, im=imethod, em=emethod, mm=mmethod)
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
        source_dump = args.dump_folder + '/' + 'source.csv'
        if os.path.isfile(source_dump):
            logger.info('loading dumped source model: {}'.format(source_dump))
            model_source = load_embeddings(folder='/', filename=source_dump, extension='', norm=True, arch='csv')[0]
        else:
            logger.info('training new source model')
            cname_source = args.cfolder + '/' + args.cname_source
            model_source = Word2Vec(size=DIM, negative=N_NS, workers=NUM_WORKERS, window=10, sg=1, null_word=1,
                                    min_count=SUB_MIN_COUNT, sample=1e-4)
            source_corpus = LineSentence(cname_source)
            model_source.build_vocab(source_corpus)
            model_source.train(source_corpus, total_examples=model_source.corpus_count,
                               epochs=model_source.epochs)
            model_source.init_sims()
            model_source = gensim2web(model_source)
            source_corpus = None
            web2csv(model_source, source_dump)
            logger.info('dumped source model as {source_dump}')
        dataset = args.vbname
        vbname = args.vbfolder + '/' + dataset
        logger.info('loading words from dataset {}'.format(dataset))
        with codecs.open(vbname, 'r', encoding='utf8', buffering=1) as fvb:
            vocab_benchmark = set(word.strip() for word in fvb)
        print vocab_benchmark

        cname_target = args.cfolder + '/' + args.cname_target
        target_corpus = LineSentence(cname_target)

        target0_dump = dump_folder + '/' + 'target0-{}.csv'.format(dataset)
        if os.path.isfile(target0_dump):
            logger.info('loading dumped target0 model: {}'.format(target0_dump))
            model_target0 = load_embeddings(folder='/', filename=target0_dump, extension='', norm=True, arch='csv')[0]
        else:
            logger.info('training new target0 model')
            model_target0 = Word2Vec(size=DIM, negative=N_NS, workers=NUM_WORKERS, window=10, sg=1, null_word=1,
                                     min_count=SUB_MIN_COUNT, sample=1e-4)
            model_target0.build_vocab(target_corpus,
                                      trim_rule=(lambda word, count, min_count:
                                                 RULE_DISCARD if word in vocab_benchmark or count < min_count
                                                              else RULE_KEEP))
            model_target0.train(target_corpus, total_examples=model_target0.corpus_count, epochs=model_target0.epochs)
            model_target0.init_sims()
            model_target0 = gensim2web(model_target0)
            web2csv(model_target0, target0_dump)
            logger.info('dumped target0 model as {}'.format(target0_dump))

        target1_dump = dump_folder + '/' + 'target1-{}.csv'.format(dataset)
        if os.path.isfile(target1_dump):
            logger.info('loading dumped target1 model: {}'.format(target1_dump))
            model_target1 = load_embeddings(folder='/', filename=target1_dump, extension='', norm=True, arch='csv')[0]
        else:
            logger.info('training new target1 model')
            vocab_target = Word2VecVocab(min_count=SUB_MIN_COUNT, null_word=1)
            vocab_target.scan_vocab(target_corpus)
            vocab_target_sorted = list(vocab_target.raw_vocab.keys())
            vocab_benchmark = vocab_benchmark.intersection(set(vocab_target_sorted))
            vocab_target_sorted.sort(key=(lambda x: vocab_target.raw_vocab[x]), reverse=False)
            size_benchmark = len(vocab_benchmark)
            for word in vocab_target_sorted:
                if word in vocab_benchmark:
                    vocab_benchmark.remove(word)
                if len(vocab_benchmark) <= size_benchmark // 2:
                    break
            model_target1 = Word2Vec(size=DIM, negative=N_NS, workers=NUM_WORKERS, window=10, sg=1, null_word=1,
                                     min_count=SUB_MIN_COUNT, sample=1e-4)
            model_target1.build_vocab(target_corpus,
                                      trim_rule=(lambda word, count, min_count:
                                                 RULE_DISCARD if word in vocab_benchmark or count < min_count
                                                              else RULE_KEEP))
            model_target1.train(target_corpus, total_examples=model_target1.corpus_count, epochs=model_target1.epochs)
            model_target1.init_sims()
            model_target1 = gensim2web(model_target1)
            web2csv(model_target1, target1_dump)
            logger.info('dumped target1 model as {}'.format(target1_dump))

        models_target = [model_target0, model_target1]
        with open('source-target.csv', 'a+') as fout:
            head = [em + im + t for t in '12' for em in 'ie' for im in 'acpz'] + \
                   ['source', 'target0', 'target1', 'dataset']
            fout.write(','.join(head) + '\n')
            ems = {'i':'intrinsic', 'd':'dataset-shadowed extrinsic'}
            ims = {'a':'global transformation', 'c': 'CCA', 'p': 'orthogonal Procrustes', 'z': 'trivial filling zeros'}
            for model_idx in (0,1):
                for em in 'id':
                    for im in 'acpz':
                        result = eval_interpolate((model_source, models_target[model_idx]), im=im, mm='c', em=em,
                                                  dataset=dataset)
                        logger.info('result of {} evaluation using approach {} is: {}'.format(ems[em], ims[im], result))
                        fout.write('{}, '.format(result))
            fout.write('{}\n'.format(dataset))

        result = 'written in source-target.csv'
    else:
        tasks = {'interpolate': interpolate, 'divide': divide_corpus}
        result = tasks[task](sys.argv[2:-1])

    with open(log_fname, 'a+') as fout:
        fout.write('configuration:\n{}\nresult:{}\n'.format(sys.argv, result))
