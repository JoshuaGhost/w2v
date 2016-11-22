from __future__ import print_function, division
EBD_DIM = 100
MIN_COUNT = 25

MODEL_FOLDER = '/tmp/w2v/custom_vocab/'
DICT_FOLDER = '/usr/share/dict/words'
CORPORA_FOLDER = '/home/assassin/workspace/master_thesis/data/sample_data'

STOPLIST = set('for a of the and to in \x00'.split())

TEST_MODE = False
NUM_ALL_DOCS = 100000
NUM_ALL_DOCS_TEST = 100
NUM_DOCS_PER_MODEL = 20000
NUM_DOCS_PER_MODEL_TEST = 20


from time import time
from copy import deepcopy
from os import path

import pickle

from gensim.models import word2vec

from utils import *

import logging
logging.basicConfig(format = '%(asctime)s : %(levelname)s : %(message)s', level = logging.INFO)
logger = logging.getLogger(__name__)

def gen_sentences_from(data_folder):
	enum_files = enumerate(walk_all_files(data_folder))
	for file_num, filename in enum_files:
		with open(filename) as f:
			yield read_sentences_from(f, STOPLIST)
		if (file_num == NUM_ALL_DOCS-1) or (TEST_MODE and file_num == NUM_ALL_DOCS_TEST-1):
			return	


model = None
if path.exists(MODEL_FOLDER + 'model_with_only_vocab.w2v') and not TEST_MODE:
	model = word2vec.Word2Vec.load(MODEL_FOLDER + 'model_with_only_vocab.w2v')
else:
	vocab_set = vocab_set_from_dict(DICT_FOLDER)
	model = word2vec.Word2Vec( size=EBD_DIM, window=10, max_vocab_size=100000,
							   sample=1e-4, workers=16, sg=1, negative=25,
							   iter=15, null_word=1, batch_words = 500)
	model.raw_vocab = {}
	model.corpus_count = 0

	sentences = []
	sent_gen = gen_sentences_from(CORPORA_FOLDER)
	for sent_idx, sentences in enumerate(sent_gen):
		scan_vocab_custom(model, sentences, vocab_set, logger)
		if (sent_idx+1)%2000 == 0:
			logger.info('collected '+str(sent_idx+1)+' sentences with '+str(len(model.raw_vocab))+' raw_vocab')
	model.scale_vocab(min_count=MIN_COUNT, trim_rule=tr_rule)
	model.finalize_vocab()
	model.save(MODEL_FOLDER + 'model_with_only_vocab.w2v')


sent_gen = gen_sentences_from(CORPORA_FOLDER)
sentences = []
for doc_idx, sentences_cur in enumerate(sent_gen):
	model_num = (doc_idx+1)//(NUM_DOCS_PER_MODEL_TEST if TEST_MODE else NUM_DOCS_PER_MODEL) 
	sentences += sentences_cur
	if ((doc_idx+1)%NUM_DOCS_PER_MODEL==0 or (doc_idx+1)%NUM_DOCS_PER_MODEL_TEST==0 and TEST_MODE):
		model_tmp = deepcopy(model)
		print (len(sentences))
		model_tmp.train(sentences, total_examples=len(sentences))
		model_tmp.save(MODEL_FOLDER + 'sub_model_ebd_100_min_count_25_'+str(model_num)+'.w2v')
		model_tmp = None
		sentences = []
