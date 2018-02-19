#!/usr/bin/python

from scalable_learning.missing_fix.interpolate import  linear_transform, fill_zero, tcca, gcca
from scalable_learning.missing_fix.evaluate_pair import eval_extrinsic, eval_intrinsic
from scalable_learning.merge import pca, concat, lra
from scalable_learning.utils import load_embeddings
import sys


def eval_interpolate(argvs):
    count_subs, subs_folder, filename, extension, im, em, mm = argvs
    subs = load_embeddings(subs_folder, filename=filename, extension=extension, norm=True, arch='csv')
    count_subs = int(count_subs)
    eval_methods = {'i': eval_intrinsic, 'e': eval_extrinsic}
    interpolate_methods = {'l': linear_transform, 't': tcca, 'g': gcca, 'z': fill_zero}
    merge_methods = {'p': pca, 'c': concat, 'l': lra}
    try:
        assert count_subs == 2
    except AssertionError:
        print("not implemented!")
        return - 1
    result = eval_methods[em](subs[:2], interpolate_methods[im], merge_methods[mm], )
    return result


def interpolate(argvs):
    pass


if __name__ == '__main__':
    '''
    python main.py operation num_subs 
        sub_folder sub_filename sub_extension 
        interpolation_method evaluation_method merge_method 
        log_file_name
    '''
    op = sys.argv[1]
    work = {'e': eval_interpolate, 'i': interpolate}
    result = work[op](sys.argv[2:-1])
    with open(sys.argv[-1], 'a+') as fout:
        fout.write('configuration:\n{}\nresult:{}\n'.format(sys.argv, result))
