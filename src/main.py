#!/usr/bin/python

from scalable_learning.missing_fix.interpolate import affine_transform, fill_zero, orthogonal_procrustes, cca
from scalable_learning.missing_fix.evaluate_pair import interpolate_combine, eval_intrinsic, eval_extrinsic, eval_demand
from scalable_learning.merge import pca, concat, lra
from scalable_learning.utils import load_embeddings, read_wv_csv, gensim2web, web2csv
from scalable_learning.corpus_divider import DividedLineSentence
from scalable_learning.extrinsic_evaluation.web.evaluate import evaluate_on_all

from gensim.models.word2vec import Word2Vec, Word2VecVocab
from gensim.utils import RULE_DISCARD, RULE_KEEP

import argparse
import sys
import logging
import codecs
import os.path

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
logging.root.setLevel(level = logging.INFO)
logger.info("running %s" % ' '.join(sys.argv))

#DIM = 500
DIM = 5
N_NS = 5
MIN_COUNT = 100
#NPARTS = 10
NPARTS = 100
SUB_MIN_COUNT = 100/NPARTS


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
            fout.write('?'.join(output_lines)+'\n')
            logger.info('sub-corpus No. {} saved'.format(idx))
    return output_fname 


if __name__ == '__main__':
    '''
    python main.py operation num_subs 
        sub_folder sub_filename sub_extension 
        interpolation_method evaluation_method merge_method 
        log_file_name
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', help='type of task')
    parser.add_argument('-l', help='logging file', default='./log.txt')
    parser.add_argument('-c', help='count of the sub models', type=int, default=2)
    parser.add_argument('-d', help='input directory')
    parser.add_argument('-f', help='(prefix of) filenames')
    parser.add_argument('-T', help='type of inputfile')
    parser.add_argument('-e', help='extention of the filenames', default='')
    parser.add_argument('--imethod', help='interpolation method a/c/p/z', default='z')
    parser.add_argument('--emethod', help='evaluation method', default='i')
    parser.add_argument('--mmethod', help='merging method', default='c')
    parser.add_argument('--cfolder', help='the folder where the sub-corpora file is saved', default='')
    parser.add_argument('--cname_source', help='filename of source sub-corpus', default='')
    parser.add_argument('--source_dump', help='directory where the source embedding is dumped', default='')
    parser.add_argument('--cname_target', help='filename of target sub-corpus', default='')
    parser.add_argument('--vbfolder', help='folder where vocabularies of benchmarks are saved', default='')
    parser.add_argument('--vbname', help='filename of benchmark vocabulary', default='')
    args = parser.parse_args()

    task, log_fname = args.t, args.l
    if task == "evaluate_2_interpolation":
        count_subs = args.c
        subs_folder, filename, extension, input_type = args.d, args.f, args.e, args.T
        imethod, emethod, mmethod = args.imethod, args.emethod, args.mmethod
        logger.info("loadding {} embeddings from {}".format(count_subs, subs_folder))
        subs = load_embeddings(subs_folder, filename=filename, extension=extension, norm=True, arch=input_type)
        result = eval_interpolate(webs=subs, im=imethod, em=emethod, mm=mmethod)
        result['num_sub'] = 2
        with open('results-{}.csv'.format(imethod), 'a+') as fout:
            fout.write(result.to_csv())

    elif task == "evaluate_single_sub":
        model_folder, filename, input_type = args.d, args.f, args.T
        fname = model_folder+'/'+filename
        
        model = read_wv_csv(fname)
        result = evaluate_on_all(model, cosine_similarity=False)
        result['num_sub'] = 1
        with open('result.csv', 'a+') as fout:
            fout.write(result.to_csv())

    elif task == 'mask_benchmark':
        source_dump = args.source_dump
        model_target2 = Word2Vec(size=DIM, negative=N_NS, workers=18, window=10, sg=1, null_word=1,
                                 min_count=SUB_MIN_COUNT, sample=1e-4)
        if os.path.isfile(source_dump):
            model_source = load_embeddings('/', source_dump, '', norm=True, arch='csv')[0]
        else:
            cname_source = args.cfolder + '/' + args.cname_source
            model_source = Word2Vec(size=DIM, negative=N_NS, workers=18, window=10, sg=1, null_word=1,
                                    min_count=SUB_MIN_COUNT, sample=1e-4)
            source_corpus = codecs.open(cname_source, 'r', encoding='utf8', buffering=1)
            model_source.build_vocab(word.split() for word in source_corpus)
            source_corpus.seek(0)
            model_source.train((word.split() for word in source_corpus), total_examples=model_source.corpus_count,
                               epochs=model_source.epochs)
            model_source.init_sims()
            model_source = gensim2web(model_source)
            source_corpus.close()
            web2csv(model_source, source_dump)

        vbname = args.vbfolder + '/' + args.vbname
        with codecs.open(vbname, 'r', encoding='utf8', buffering=1) as fvb:
            vocab_benchmark = set(word for word in fvb)
        cname_target = args.cfolder + '/' + args.cname_target
        target_corpus = codecs.open(cname_target, 'r', encoding='utf8', buffering=1)
        model_target1 = Word2Vec(size=DIM, negative=N_NS, workers=18, window=10, sg=1, null_word=1,
                                 min_count=SUB_MIN_COUNT, sample=1e-4)
        model_target1.build_vocab((word.split() for word in target_corpus),
                                  trim_rule=(lambda word, count, min_count:
                                             RULE_DISCARD if word in vocab_benchmark else RULE_KEEP))
        target_corpus.seek(0)
        model_target1.train((word.split() for word in target_corpus), total_examples=model_target1.corpus_count,
                            epochs=model_target1.epochs)
        model_target1.init_sims()
        model_target1 = gensim2web(model_target1)

        target_corpus.seek(0)
        vocab_target = Word2VecVocab(min_count=1, null_word=1)
        vocab_target.scan_vocab(word.split() for word in target_corpus)
        vocab_target_sorted = list(vocab_target.raw_vocab.keys())
        vocab_benchmark = vocab_benchmark.intersection(set(vocab_target_sorted))
        vocab_target_sorted.sort(key=(lambda x: vocab_target.raw_vocab[x]), reverse=True)
        size_benchmark = len(vocab_benchmark)
        for word in vocab_target_sorted:
            if word in vocab_benchmark:
                vocab_benchmark.remove(word)
            if len(vocab_benchmark) <= size_benchmark//2:
                break
        target_corpus.seek(0)
        model_target2.build_vocab((word.split() for word in target_corpus),
                                  trim_rule=(lambda word, count, min_count:
                                             RULE_DISCARD if word in vocab_benchmark else RULE_KEEP))
        target_corpus.seek(0)
        model_target2.train((word.split() for word in target_corpus), total_examples=model_target2.corpus_count,
                            epochs=model_target2.epochs)
        model_target2.init_sims()
        model_target2 = gensim2web(model_target2)

        results_intrinsic1 = {}
        results_intrinsic2 = {}
        results_extrinsic1 = {}
        results_extrinsic2 = {}
        dataset = args.vbname

        for im in 'acpz':
            #import pdb
            #pdb.set_trace()
            results_intrinsic1[im] = eval_interpolate((model_source, model_target1), im=im, mm='c', em='i')
            results_extrinsic1[im] = eval_interpolate((model_source, model_target1), im=im, mm='c', em='d',
                                                      dataset=dataset)
            results_intrinsic2[im] = eval_interpolate((model_source, model_target1), im=im, mm='c', em='i')
            results_extrinsic2[im] = eval_interpolate((model_source, model_target2), im=im, mm='c', em='d',
                                                      dataset=dataset)
        result_extrinsic_source = eval_demand(model_source, dataset=dataset)
        result_extrinsic_target1 = eval_demand(model_target1, dataset=dataset)
        result_extrinsic_target2 = eval_demand(model_target2, dataset=dataset)
        with open('source-target.csv', 'a+') as fout:
            head = [em+im+t for t in '12' for em in 'ie' for im in 'acpz'] + ['source', 'target1', 'target2', 'dataset']
            fout.write(','.join(head)+'\n')
            for im in 'acpz':
                fout.write('{}, '.format(results_intrinsic1[im]))
            for im in 'acpz':
                fout.write('{}, '.format(results_extrinsic1[im]))
            for im in 'acpz':
                fout.write('{}, '.format(results_intrinsic2[im]))
            for im in 'acpz':
                fout.write('{}, '.format(results_extrinsic2[im]))
            fout.write('{}, {}, {}, {}\n'.format(result_extrinsic_source,
                                                 result_extrinsic_target1,
                                                 result_extrinsic_target2,
                                                 dataset))
            result = 'written in source-target.csv'
    else:
        tasks = {'interpolate': interpolate, 'divide':divide_corpus}
        result = tasks[task](sys.argv[2:-1])

    with open(log_fname, 'a+') as fout:
        fout.write('configuration:\n{}\nresult:{}\n'.format(sys.argv, result))
