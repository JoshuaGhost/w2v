from gensim.models.word2vec import Word2Vec
from pathlib2 import Path
from pickle import load, dump
from numpy import log, exp, vstack, hstack, array
import logging
import numpy as np

tmp_dir = Path('/tmp/zzj').resolve()

Ws_dump_dir = tmp_dir/Path('Ws.pkl')
Cs_dump_dir = tmp_dir/Path('Cs.pkl')
counts_dump_dir = tmp_dir/Path('counts.pkl')
vocab_common_dump_dir = tmp_dir/Path('vocab_common.pkl')

output_dir = tmp_dir/Path('combined_new.pkl')

FORMAT = "%(asctime)-15s %(message)s"
logging.basicConfig(format=FORMAT)
logger = logging.getLogger('combining_document_new')
logger.setLevel(logging.INFO)
subs_dir = Path('../../models/documentwise_100/subs/').resolve()

dim_merge = 500

try:
    Ws = load(open(str(Ws_dump_dir)))
    Cs = load(open(str(Ws_dump_dir)))
    counts = load(open(str(counts_dump_dir)))
    vocab_common = load(open(str(vocab_common_dump_dir)))
except IOError:
    Ws = []
    Cs = []
    vocabs = []
    counts = []
    word2indexes = []

    for idx, model_name in enumerate(subs_dir.glob("sub.??")):
        m = Word2Vec.load(str(model_name))
        wc = dict([[word, m.wv.vocab[word].count] for word in m.wv.vocab.keys()])
        vocab_size = len(m.wv.index2word)
        vocab = []
        count = []
        W = []
        C = []
        word2index = {}
        for word_index in xrange(vocab_size):
            word = m.wv.index2word[word_index]
            vocab.append(word)
            count.append(m.wv.vocab[word].count)
            W.append(m.wv.syn0[word_index])
            C.append(m.syn1neg[word_index])
            word2index[word] = word_index
        vocabs.append(vocab)
        counts.append(count)
        Ws.append(W)
        Cs.append(C)
        word2indexes.append(word2index)

    vocab_common = list(reduce(lambda x,y: x.intersection(y), [set(vocab) for vocab in vocabs]))
    #word2index_common = dict([[word, idx] for idx, word in enumerate(vocab_common)])
    num_models = len(Ws)

    for model_idx in xrange(num_models):
        W = []
        count = []
        C = []
        for word_new_idx, word in enumerate(vocab_common):
            word_idx = word2indexes[model_idx][word]
            W.append(Ws[model_idx][word_idx])
            C.append(Cs[model_idx][word_idx])
            count.append(counts[model_idx][word_idx])
        Ws[model_idx] = W
        Cs[model_idx] = C
        counts[model_idx] = count
    
    Ws = array(Ws)
    Cs = array(Cs)
    counts = array(counts)
    dump(Ws, open(str(Ws_dump_dir), 'w+'))
    dump(Cs, open(str(Cs_dump_dir), 'w+'))
    dump(counts, open(str(counts_dump_dir), 'w+'))
    dump(vocab_common, open(str(vocab_common_dump_dir), 'w+'))

num_models = len(Ws)
num_vocab = len(vocab_common)
logger.info("{} sub-models loaded, {} words in common vocabulary".format(num_models, len(vocab_common)))

logger.info("word-wise aligned, start to merging")

iteration = 10
Wn = np.random.random((num_vocab, dim_merge))
Cn = np.random.random((dim_merge, num_vocab))

log_count_total = log(counts.sum(axis=0))

def approach(X, Y, G):
    learning_rat = 0.001
    threshold = 0.0001
    while True:
        err = X.dot(Y) - G
        if max(abs(err)) < threshold:
            return X, Y
        X -= learning_rate * err.dot(Y.T)
        Y -= learning_rate * X.T.dot(err)

for i in xrange(iteration):
    for idxw, word in enumerate(vocab_common):
        temp = [exp(-Ws[idxm, idxw].dot(Cs[idxm])) for idxm in xrange(num_models)]
        print temp.shape
        print counts.shape
        print count[0].shape
        print count[0, idxw].shape
        temp = [temp*counts[idxm,idxw]*counts[idxm] for idxm in xrange(num_models)]
        temp = temp.sum(axis=0)
        lhs = -log(temp*num_models**2)
        goal = lhs+log_count_total[idxw]+log_count_total
        Wn[idxw], Cn = approach(Wn[idxw], Cn, goal)

dump(dict(zip(vocab_common, Wn)), open(str(output_dir)))
