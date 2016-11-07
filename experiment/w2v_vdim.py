import numpy
import pickle
import logging
logging.basicConfig(format = '%(asctime)s : %(levelname)s : %(message)s', level = logging.DEBUG)
logger = logging.getLogger(__name__)

from utils import *
from config_vdim import *

from gensim.models import Word2Vec


for ebd_dim in range(ebd_dim_begin, ebd_dim_end, ebd_dim_step):

	model = Word2Vec(iter = 15, size = ebd_dim, workers = 24, sample = 1e-4, negative = 25, batch_words = 500)
	word2vec_vvocab_name = '/tmp/word2vec_vvocab'+str(vocab_freq_thres) + '_ebd' +str(ebd_dim) + '.txt'

	logger.info("model will be saved at "+word2vec_vvocab_name)

	if os.path.exists(word2vec_vvocab_name):
		continue

	if os.path.exists(raw_sentences):
		with open(raw_sentences) as ifile:
			sentences = pickle.load(ifile)
			model.scan_vocab(sentences)
			logger.info("retrieved "+str(len(sentences))+" raw sentences from "+raw_sentences)

	else:
		sentences = []
		for filename in walk_all_files(data_folder):
			with open(filename) as f:
				sentences += read_sentences_from(f, prune_list = stoplist)
			if test_mode and len(sentences) > len_sentc_test:
				break
		with open(raw_sentences, 'w') as ofile:
			pickle.dump(sentences, ofile)
		logger.info("raw sentences saved in "+raw_sentences)
		model.scan_vocab(sentences)
		
	model.scale_vocab(keep_raw_vocab = True, min_count = vocab_freq_thres)
	logger.info("length of vocabulary list: "+str(len(model.vocab)))
	model.finalize_vocab()


	docs_gathered = 0
	sentences = []
	for filename in walk_all_files(data_folder):
		docs_gathered +=1
		sentences_tmp = []
		with open(filename) as f:
			sentences_tmp = read_sentences_from(f, prune_list = stoplist)
		if len(sentences) >= num_sentences_per_batch:
			model.train(sentences)
			sentences = []
			logger.info(str(docs_gathered)+" docs have been processed")

		if test_mode and docs_gathered > numdocs_test_batch:
			for (n, k) in enumerate(model.vocab):
				if n > 5:
					break
				print k, ': ', model.most_similar(positive = [k], topn = 3)
			model.save(word2vec_vvocab_name+'.test')
			print str(model.vocab['life,\xe2\x80\x9d'])
			print str(model.vocab['\xef\xbf\xbdwe'])
			print docs_gathered
			break

	if not test_mode:
		model.save(word2vec_vvocab_name)

