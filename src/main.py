from missing_fix.evaluation.evaluate_pair import *
from missing_fix.interpolate import *
import sys

def eval_interpolate(argvs):
    count_subs, em, im = argvs[0:3]
    subs_folder = argvs[2]
    subs = load_models_from(subs_folder)
    count_subs = int(count_subs)
    eval_methods = {'i': eval_intrinsic, 'e': eval_extrinsic}
    interpolate_methods = {'l': linear_transformation, 't': tensor_cca, 'g': gcca}
    try:
        assert count_subs == 2
    except AssertionError:
        print ("not implemented!")
        return -1
    result = eval_methods[em](subs[:2], interpolate_methods[im])
    return result
        
def interpolate(argvs):
    pass

if __name__ == '__main__':
    op = sys.argv[1]
    work = {'e': eval_interpolate, 'i': interpolate}
    result = work[op](sys.argv[2:])
    with open(sys.argv[-1], 'a+') as fout:
        fout.write('configuration:\n{}\nresult:{}\n\n'.format(sys.argv, result))
