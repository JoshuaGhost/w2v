NUM_SENTENCES_PER_BATCH = 100000

DATA_FOLDER = '/home/assassin/workspace/data'
#DATA_FOLDER = '/home/zijian/workspace/data'
RES_FOLDER = '/tmp/w2v/'
RAW_DATA_DUMP_FOLDER = RES_FOLDER

STOPLIST = set('for a of the and to in \x00'.split())

TEST_MODE = True
NUM_ALL_DOCS_TEST_MODE = 100
NUM_DOCS_PER_MODEL_TEST_MODE = 20

NUM_ALL_DOCS = 50000
NUM_DOCS_PER_MODEL = 10000


EBD_DIM = 100
VOCAB_FREQ_THRES = 50