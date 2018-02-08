import logging
import sys
import os
import pickle
from web.evaluate_on_demand import evaluate_words_demanded_on_all

program = os.path.basename(sys.argv[0])
logger = logging.getLogger(program)
logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
logging.root.setLevel(level=logging.INFO)

model_name = sys.argv[1].split('/')[-1].split('.')[0]

if len(sys.argv)>2:
    with open(sys.argv[2]) as f:
        words_demanded = [word.strip().decode('utf-8') for word in f.readlines()]
        vocab_name = sys.argv[2].split('/')[-1].split('.')[0]
else:
    words_demanded = None
    vocab_name = None

cosine_similarity = False

w = pickle.load(open(sys.argv[1], 'r'))
logger.info('[NEW]Start to evaluate {} on vocabulary {}'.format(model_name, vocab_name))
result = evaluate_words_demanded_on_all(w, cosine_similarity, words_demanded)

result.to_csv(model_name +
              ('' if vocab_name is None else vocab_name) +
              '.csv')
