from __future__ import print_function
from utils import *
from config import *
from time import time
from gensim.models import Word2Vec


import numpy
import pickle
import logging
logging.basicConfig(format = '%(asctime)s : %(levelname)s : %(message)s', level = logging.INFO)
logger = logging.getLogger(__name__)


stoplist = set('for a of the and to in \x00'.split())


timestamp = -time()
dictionary = vocab_set_from_dict(DICT_PATH)
sentences = []
for sen in gen_sentence(CORPORA_DIR, NUM_DOCS, stoplist, dictionary, test_mode = TEST_MODE, num_docs_test = NUM_DOCS_TEST):
	sentences += sen
timestamp += time()
with open('time.txt', 'w+') as f:
	f.write('sentences collection time:\n %0.6f\n' % timestamp)


timestamp = -time()
model = Word2Vec(sentences = sentences, size = EBD_DIM, window = 10, min_count = MIN_COUNT,
		max_vocab_size = MAX_VOCAB_SIZE, sample = 1e-4, workers = WORKERS, 
		sg = 1, negative = 25, iter = ITER, null_word = 1,
		trim_rule = tr_rule, batch_words = BATCH_WORDS)
model_filename = 'ebd_100_mc_25_regenerated.w2v'
if TEST_MODE:
	model_filename += '.test'
model.save(MODELS_DIR + model_filename)
timestamp += time()
with open('time.txt', 'a+') as f:
	f.write("training and saving time:\n %0.6f\n" % timestamp)
