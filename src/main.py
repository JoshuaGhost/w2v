#!/usr/bin/python

from scalable_learning.missing_fix.interpolate import affine_transform, fill_zero, orthogonal_procrustes, cca
from scalable_learning.missing_fix.evaluate_pair import eval_extrinsic, eval_intrinsic
from scalable_learning.merge import pca, concat, lra
from scalable_learning.utils import load_embeddings
from scalable_learning.corpus_divider import DividedLineSentence

import argparse
import sys, logging
from time import localtime, strftime

logger = logging.getLogger(__name__)
logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
logging.root.setLevel(level = logging.INFO)
logger.info("running %s" % ' '.join(sys.argv))

def eval_interpolate(count_subs, subs_folder, filename, extension, im, em, mm):
    logger.info("loadding {} embeddings from {}".format(count_subs, subs_folder))
    subs = load_embeddings(subs_folder, filename=filename, extension=extension, norm=True, arch='csv')
    count_subs = int(count_subs)
    try:
        assert count_subs==subs
    except AssertionError:
        logger.error('No. embedding loaded is not the same with designated count!')
        retur -1
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
    parser.add_argument('-e', help='extention of the filenames', default='')
    parser.add_argument('--im', help='interpolation method', default='z')
    parser.add_argument('--em', help='evaluation method', default='i')
    parser.add_argument('--mm', help='merging method', default='c')
    args = parser.parse_args()
    task = args.w
    log_fname = args.l
    if task == "evaluate_interpolation":
        count_subs, subs_folder, filename, extension, im, em, mm = args.c, args.d, args.f, args.e, args.im, args.em, args.mm
        result = eval_interpolate(cout_subs, subs_folder, filename, extension, im, em, mm)
    else:
        tasks = {'interpolate': interpolate, 'divide':divide_corpus}
        result = tasks[t](sys.argv[2:-1])
    with open(log_fname, 'a+') as fout:
        fout.write('configuration:\n{}\nresult:{}\n'.format(sys.argv, result))

