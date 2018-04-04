#!/usr/bin/python

from scalable_learning.missing_fix.interpolate import affine_transform, fill_zero, orthogonal_procrustes, cca
from scalable_learning.missing_fix.evaluate_pair import eval_extrinsic, eval_intrinsic
from scalable_learning.merge import pca, concat, lra
from scalable_learning.utils import load_embeddings, read_wv_csv
from scalable_learning.corpus_divider import DividedLineSentence
from scalable_learning.extrinsic_evaluation.web.evaluate import evaluate_on_all
from gensim.models.word2vec import LineSentence, Word2Vec, Word2VecVocab
from gensim.utils import RULE_DISCARD, RULE_KEEP
import argparse
import sys, logging
import codecs
from time import localtime, strftime

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
logging.root.setLevel(level = logging.INFO)
logger.info("running %s" % ' '.join(sys.argv))

DIM = 500
N_NS = 5
MIN_COUNT = 100
NPARTS = 100
SUB_MIN_COUNT = 100/NPARTS

def eval_interpolate(count_subs, subs_folder, filename, extension, im, em, mm, input_type):
    logger.info("loadding {} embeddings from {}".format(count_subs, subs_folder))
    subs = load_embeddings(subs_folder, filename=filename, extension=extension, norm=True, arch=input_type)
    count_subs = int(count_subs)
    try:
        assert count_subs == len(subs)
    except AssertionError:
        logger.error('No. embedding loaded is not the same with designated count!')
        return -1
    eval_methods = {'i': eval_intrinsic, 'e': eval_extrinsic}
    interpolate_methods = {'a': affine_transform, 'p': orthogonal_procrustes, 'c': cca, 'z': fill_zero}
    merge_methods = {'p': pca, 'c': concat, 'l': lra}
    try:
        assert count_subs == 2
    except AssertionError:
        print("not implemented!")
        return -1
    logger.info('begin to run {} evaluation for interpolation method {}, the merging method is {}'.format(
                eval_methods[em].__name__, interpolate_methods[im], merge_methods[mm]))
    ret = eval_methods[em](subs[:2], interpolate_methods[im], merge_methods[mm])
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
    parser.add_argument('--cname', help='''sub-corpora filename, one sub-corpus per line.
                                           Each line is begin with sub-corpus index and a colon \':\'.
                                           Sentences are separated with question mark \'?\' ''', default='')
    parser.add_argument('--vbfolder', help='folder where vocabularies of benchmarks are saved', default='')
    parser.add_argument('--vbname', help='filename of benchmark vocabulary', default='')
    args = parser.parse_args()

    task, log_name = args.t, args.l
    if task == "evaluate_2_interpolation":
        count_subs = args.c
        subs_folder, filename, extension, input_type = args.d, args.f, args.e, args.T
        imethod, emethod, mmethod = args.imethod, args.emethod, args.mmethod

        result = eval_interpolate(count_subs, subs_folder, filename, extension, imethod, emethod, mmethod, input_type)
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
        vbfolder = args.vbfolder
        vbname = args.vbname
        corpus = args.cfolder + '/' + args.cname
        fcorpus = codecs.open(corpus, 'r', encoding='utf8', buffering=1)
        sub_corpus = [word.split() for word in fcorpus.readline().split(':')[1:].split('?')]
        model_source = Word2Vec(sub_corpus, size=DIM, negative=N_NS, workers=18, window=10, sg=1,
                                null_word=1, min_count=SUB_MIN_COUNT, sample=1e-4)
        vt = Word2VecVocab(min_count=1, null_word=0)
        sub_corpus = [word.split() for word in fcorpus.readline().split(':')[1:].split('?')]
        vt.scan
    else:
        tasks = {'interpolate': interpolate, 'divide':divide_corpus}
        result = tasks[t](sys.argv[2:-1])
    with open(log_fname, 'a+') as fout:
        fout.write('configuration:\n{}\nresult:{}\n'.format(sys.argv, result))
