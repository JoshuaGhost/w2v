EBD_DIM_DEFAULT = 100
MIN_COUNT_DEFAULT = 25

MODELS_FOLDER 		= '/tmp/w2v/default_models/'
CORPORA_FOLDER 		= '/home/zijian/workspace/data/'
BENCHMARK_FOLDER 	= ''
CE_FOLDER			= '/tmp/w2v/default_results/'
DICT_PATH	 		= '/usr/share/dict/words'

STOPLIST = set('for a of the and to in \x00'.split())

TEST_MODE = False
NUM_ALL_DOCS = 100000
NUM_ALL_DOCS_TEST = 100

MIN_COUNT = 24
MIN_COUNT_MIN = 0
MIN_COUNT_MAX = 50
MIN_COUNT_STEP = 2

DIM = 100
DIM_MIN = 25
DIM_MAX = 300
DIM_STEP = 25

SENTENCES_PER_BATCH = 100000

raw_sentences = 'raw_sentences.txt'

stoplist = set('for a of the and to in \x00'.split())

MAX_VOCAB_SIZE = 10000000
NUM_WORKERS = 16
NUM_ITER = 15
BATCH_WORDS = 500

NUM_SUB_MODEL_DEFAULT = 5