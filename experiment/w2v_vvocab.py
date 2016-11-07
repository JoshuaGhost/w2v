import numpy
import pickle
import logging
logging.basicConfig(format = '%(asctime)s : %(levelname)s : %(message)s', level = logging.INFO)
logger = logging.getLogger(__name__)

from utils import *
from config_vvocab import *

from gensim.models import Word2Vec

docs_gathered = 0
sentences = []
for filename in walk_all_files(data_folder):
	docs_gathered += 1
	with open(filename) as f:
		sentences += read_sentences_from(f, prune_list = stoplist)
	if test_mode and len(sentences) > len_sentc_test:
		break
	if docs_gathered >= num_all_docs:
		break
	if docs_gathered % 10000 == 0:
		logger.info(str(docs_gathered) + " docs have been gathered")
logger.info("sentences gathering finished")
logger.info("in total " + str(docs_gathered) + " docs have been gathered")


for vocab_freq_thres in range(vocab_freq_thres_begin, vocab_freq_thres_end, vocab_freq_thres_step):

	word2vec_vvocab_name = '/tmp/word2vec_vvocab'+str(vocab_freq_thres) + '_ebd' +str(ebd_dim) + '.txt'

	if os.path.exists(word2vec_vvocab_name):
                continue

	logger.info("model will be saved at "+word2vec_vvocab_name)

	model = Word2Vec(sentences = sentences, size = ebd_dim, window = 10, min_count = vocab_freq_thres,
			 max_vocab_size = 10000000, sample = 1e-4, workers = 24, 
			 sg = 1, negative = 25, iter = 15, null_word = 1,
			 trim_rule = tr_rule, batch_words = 500)
	
	if test_mode:
		for (n, k) in enumerate(model.vocab):
			if n > 5:
				break
			print k, ': ', model.most_similar(positive = [k], topn = 3)
		model.save(word2vec_vvocab_name+'.test')
		continue

	if not test_mode:
		model.save(word2vec_vvocab_name)

