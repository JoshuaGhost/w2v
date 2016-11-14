from __future__ import print_function, division
import logging
import pickle
from time import time
logging.basicConfig(format = '%(asctime)s : %(levelname)s : %(message)s', level = logging.INFO)
logger = logging.getLogger(__name__)

from utils import *
from config import *

from gensim.models import Word2Vec

docs_gathering_times = []
docs_gathered = 0
sentences = []

raw_data_dump_file = 'raw_data_for_model_'

numbered_files = enumerate(walk_all_files(DATA_FOLDER))
docs_gathering_times.append(-time())
for file_num, filename in numbered_files:
	if (file_num > 0) and (len(sentences) > 0):
		if ((file_num % NUM_DOCS_PER_MODEL_TEST_MODE == 0) and TEST_MODE) or\
			(file_num % NUM_DOCS_PER_MODEL == 0):
			pickle.dump(sentences, open(RAW_DATA_DUMP_FOLDER + raw_data_dump_file, 'w'))
			sentences = []
			docs_gathering_times[-1]+=time()
			docs_gathering_times.append(-time())
	
	if (TEST_MODE and file_num > NUM_ALL_DOCS_TEST_MODE) or (file_num > NUM_ALL_DOCS):
		if len(sentences)>0:
			pickle.dump(sentences, open(RAW_DATA_DUMP_FOLDER + raw_data_dump_file, 'w'))
			sentences = []
		docs_gathering_times[-1]+=time()
		break

	if file_num % 2000 == 0:
		logger.info(str(file_num) + " docs have been gathered")
	
	model_num = str(file_num // (NUM_DOCS_PER_MODEL_TEST_MODE if TEST_MODE else NUM_DOCS_PER_MODEL))
	raw_data_dump_file = 'raw_data_multi_models_no_' + str(model_num) + '.txt'
	
	if os.path.exists(RAW_DATA_DUMP_FOLDER + raw_data_dump_file) and not TEST_MODE:
		continue
	
	with open(filename) as f:
		sentences += read_sentences_from(f, prune_list = STOPLIST)

logger.info("doc gathering complete")
with open("times.csv", "w+") as f:
	f.write('documentations gathering time:\n')
	f.write('\n'.join(map(str, docs_gathering_times))+'\n')


training_times = []
num_models = (NUM_ALL_DOCS_TEST_MODE // NUM_DOCS_PER_MODEL_TEST_MODE)\
				if TEST_MODE\
				else (NUM_ALL_DOCS // NUM_DOCS_PER_MODEL)

for num_model in range(num_models):
	
	training_times.append(-time())
	model_file_name = 'multi_models_ebd_' + str(EBD_DIM) + '_vvth_' + str(VOCAB_FREQ_THRES) + '_no_' + str(num_model)+'.w2v'
	
	if os.path.exists(RES_FOLDER + model_file_name) and not TEST_MODE:
		continue

	raw_data_dump_file = 'raw_data_multi_models_no_' + str(num_model) + '.txt'
	
	TEST_MODE and logger.info("under test mode")
	logger.info("model will be saved at " + RES_FOLDER + model_file_name)

	logger.info('retreave dumped sentences from ' + RAW_DATA_DUMP_FOLDER + raw_data_dump_file)
	sentences = pickle.load(open(RAW_DATA_DUMP_FOLDER + raw_data_dump_file))

	model = Word2Vec(sentences = sentences, size = EBD_DIM, window = 10, min_count = VOCAB_FREQ_THRES,
		max_vocab_size = 10000000, sample = 1e-4, workers = 16, 
		sg = 1, negative = 25, iter = 15, null_word = 1,
		trim_rule = tr_rule, batch_words = 500)
		
	if TEST_MODE:
		for (n, k) in enumerate(model.vocab):
			if n > 3: break
			print (k, ': ', model.most_similar(positive = [k], topn = 3))
			model.save(RES_FOLDER + model_file_name + '.test')
		print('=============================')

	else:
		model.save(RES_FOLDER + model_file_name)
	training_times[-1] += time()

with open("times.csv", "a+") as f:
	f.write('models training time:\n')
	f.write('\n'.join(map(str, training_times))+'\n')
