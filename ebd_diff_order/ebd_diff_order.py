from __future__ import print_function
import logging
import pickle
import time
logging.basicConfig(format = '%(asctime)s : %(levelname)s : %(message)s', level = logging.INFO)
logger = logging.getLogger(__name__)

from utils import *
from config import *

from gensim.models import Word2Vec


docs_gathered = 0
sentences = []
if not os.path.exists(raw_dump_folder+raw_dump_file):
    time_gather_docs = -time.time()
    for filename in walk_all_files(data_folder):
        docs_gathered += 1
	with open(filename) as f:
	    sentences += read_sentences_from(f, prune_list = stoplist)
	    
            if docs_gathered % 5000 == 0:
	        logger.info(str(docs_gathered) + " docs have been gathered")
	    
            if (test_mode and docs_gathered >= numdocs_test_batch) or (docs_gathered >= num_all_docs):
                time_gather_docs += time.time()
                pickle.dump(sentences, open(raw_dump_folder+raw_dump_file, 'w'))
                break

    logger.info("doc gathering complete")
    logger.info("in total " + str(docs_gathered) + " docs have been gathered")

else:
    logger.info('retreave dumped sentences from ' + raw_dump_folder + raw_dump_file)
    sentences = pickle.load(open(raw_dump_folder + raw_dump_file))


ebd_dim = 100
vocab_freq_thres = 50

otypes = ['orig', 'rev', 'rand']

sentences_in_order = {}
sentences_in_order['orig'] = sentences #original order
num_sentences = len(sentences_in_order['orig'])
sentences_in_order['rev'] = [s for s in reversed(sentences_in_order['orig'])] #reversed
import random
random.seed(19920523)
rand_order = [random.randint(0, num_sentences) for i in range(num_sentences)]
order_pair = dict(zip(rand_order, [i for i in range(num_sentences)]))
rand_order.sort()
sentences_in_order['rand'] = [sentences_in_order['orig'][order_pair[i]] for i in rand_order] #randomized


for otype in otypes:
    model_file_name = 'w2v_'+otype+'_order_ebd_100_vvth_50.w2v'
    if os.path.exists(res_folder + model_file_name) and not test_mode:
        continue
    
    test_mode and logger.info("under test mode")
    logger.info("model will be saved at " + res_folder + model_file_name)

    
    model = Word2Vec(sentences = sentences_in_order[otype], size = ebd_dim, window = 10, min_count = vocab_freq_thres,
        max_vocab_size = 10000000, sample = 1e-4, workers = 16, 
        sg = 1, negative = 25, iter = 15, null_word = 1,
        trim_rule = tr_rule, batch_words = 500)
		
    if test_mode:
        for (n, k) in enumerate(model.vocab):
            if n > 5: break
            print (k, ': ', model.most_similar(positive = [k], topn = 3))
            model.save(res_folder + model_file_name + '.test')

    else:
        model.save(res_folder + model_file_name)


