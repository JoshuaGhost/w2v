from __future__ import print_function
import logging
import pickle
import time
logging.basicConfig(format = '%(asctime)s : %(levelname)s : %(message)s', level = logging.INFO)
logger = logging.getLogger(__name__)

from utils import *
from config import *

from gensim.models import Word2Vec


time_gather_docs = 0.
time_retrieve_raw = 0.
time_train = [[0 for vocab_freq_thres in range(vocab_freq_thres_begin, vocab_freq_thres_end, vocab_freq_thres_step)] 
			for ebd_dim in range(ebd_dim_begin, ebd_dim_end, ebd_dim_step)]


docs_gathered = 0
if not os.path.exists(raw_dump_folder+raw_dump_file):
	sentences = []
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
logger.info("in total "+str(docs_gathered)+" docs have been gathered")


if not os.path.exists('time.csv') or test_mode:
	with open('times.csv', 'w+') as f:
		f.write("docs gathered within: "+str(time_gather_docs)+"s\n")
		f.write("embedding dimension, vocabulary stop threshold, training time\n")


for ebd_dim in range(ebd_dim_begin, ebd_dim_end, ebd_dim_step):
	for vocab_freq_thres in range(vocab_freq_thres_begin, vocab_freq_thres_end, vocab_freq_thres_step):
		
		word2vec_model_file = 'word2vec_ebd' + str(ebd_dim) + '_vvth'+ str(vocab_freq_thres) + '.w2v'
		if os.path.exists(res_folder + word2vec_model_file) and not test_mode:
			continue
				
		test_mode and logger.info("under test mode")
		logger.info("model will be saved at " + res_folder + word2vec_model_file)

		time_retrieve_raw -= time.time()
		sentences = pickle.load(open(raw_dump_folder + raw_dump_file))
		time_retrieve_raw += time.time()
		
		time_current_config = -time.time()
		model = Word2Vec(sentences = sentences, size = ebd_dim, window = 10, min_count = vocab_freq_thres,
				 max_vocab_size = 10000000, sample = 1e-4, workers = 16, 
				 sg = 1, negative = 25, iter = 15, null_word = 1,
				 trim_rule = tr_rule, batch_words = 500)
		time_current_config += time.time()
		time_train[ebd_dim / ebd_dim_step - 1][vocab_freq_thres / vocab_freq_thres_step -1] += time_current_config
		
		with open('times.csv', 'a+') as f:
			f.write(', '.join((str(ebd_dim), str(vocab_freq_thres), str(time_current_config))) + '\n')

		if test_mode:
			for (n, k) in enumerate(model.vocab):
				if n > 5:
					break
				print (k, ': ', model.most_similar(positive = [k], topn = 3))
			model.save(res_folder + word2vec_model_file+'.test')
		else:
			model.save(res_folder + word2vec_model_file)


with open('times.csv', 'a+') as f:
	f.write("average dump retrieving time: "+str(time_retrieve_raw/(len(time_train)*len(time_train[0])))+'s\n')
