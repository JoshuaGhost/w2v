ebd_dim_begin = 50
ebd_dim_end   = 301
ebd_dim_step  = 50

vocab_freq_thres_begin = 5
vocab_freq_thres_end   = 51
vocab_freq_thres_step  = 5

num_sentences_per_batch = 100000

data_folder = '/home/zijian/workspace/data'
res_folder = '/tmp/w2v/'
raw_dump_folder = res_folder
raw_dump_file = 'raw.txt'

stoplist = set('for a of the and to in \x00'.split())

test_mode = False
numdocs_test_batch = 100
len_sentc_test = 10000

num_all_docs = 100000
