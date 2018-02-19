import random
fname = 'comb_eval_i.txt'
with open(fname, 'w+') as f:
    for count_subs in range(1, 11):
        for i in range(10):
            idx_list = random.sample(range(10), count_subs)
            f.write(repr(idx_list)+'\n')

