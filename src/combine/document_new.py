from gensim.models.word2vec import Word2Vec
from pathlib2 import Path
from pickle import load, dump
from numpy import log, exp, vstack, hstack, array, zeros, ones
from scipy.special import expit
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


try:
    Ws = load(open(str(Ws_dump_dir)))
    Cs = load(open(str(Ws_dump_dir)))
    counts = load(open(str(counts_dump_dir)))
    vocab_common = load(open(str(vocab_common_dump_dir)))
    #Ws = ones((100, len(vocab_common), 50))
    #Cs = ones((100, 50, len(vocab_common)))
    #counts = ones((100, len(vocab_common)))
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
dim_old = len(Ws[0][0])

dim_merge = 500

logger.info("{} sub-models loaded, {} words in common vocabulary".format(num_models, len(vocab_common)))

logger.info("word-wise aligned, start to merging")

iteration = 10
Wn = np.random.random((num_vocab, dim_merge))/dim_merge
Cn = np.random.random((dim_merge, num_vocab))/dim_merge

log_count_sum = log(counts.sum(axis=0))

def approach(X, Y, G):
    learning_rate = 0.001
    err_threshold = 0.1
    epsilon = 1e-9
    beta1 = 0.9
    beta2 = 0.9999

    mX = zeros((1, dim_merge))
    mY = zeros((dim_merge, num_vocab))

    VX = zeros((1, dim_merge))+epsilon
    VY = zeros((dim_merge, num_vocab))+epsilon

    X = X.reshape((1, dim_merge))
    Y = Y.reshape((dim_merge, num_vocab))

    G = G.reshape((1, num_vocab))
    G = expit(G)

    while True:
        P = expit(X.dot(Y))
        E = (P - G)
        max_E = max(abs(E.reshape(num_vocab))) 
        
        if max_E < err_threshold:
            return X, Y
        
        logger.info('max error: {}'.format(max_E))
        
        Xahead = X-learning_rate*mX/VX
        Yahead = Y-learning_rate*mY/VY
        
        Pahead = expit(Xahead.dot(Yahead))
        Eahead = Pahead-G
        gradient = Eahead*Pahead*(1-Pahead)
        gradientX = gradient.dot(Y.T)
        gradientY = X.T.dot(gradient)

        mX = beta1*mX+(1-beta1)*gradientX
        mY = beta1*mY+(1-beta1)*gradientY
        
        VX = beta2*VX+(1-beta2)*gradientX**2+epsilon
        VY = beta2*VY+(1-beta2)*gradientY**2+epsilon
        
        X -= learning_rate * mX/VX
        Y -= learning_rate * mY/VY

for i in xrange(iteration):
    for idxw, word in enumerate(vocab_common):
        temp = []
        for idxm in xrange(num_models):
            w = Ws[idxm, idxw].reshape((1, dim_old))
            cs = Cs[idxm].reshape((dim_old, num_vocab))
            temp.append(exp(-w.dot(cs)))
        temp = array(temp)
        for idxm in xrange(num_models):
            count_w = counts[idxm, idxw]
            counts_c = counts[idxm].reshape((1, num_vocab))
            temp[idxm] *= count_w * counts_c
        temp = temp.sum(axis=0)
        lhs = -log(temp*num_models**2)
        goal = lhs+log_count_sum[idxw]+log_count_sum
        Wn[idxw], Cn = approach(Wn[idxw], Cn, goal)
        logger.info("word idx {} learning complete".format(idxw))
    logger.info('iteration {} complete'.fomat(i))

dump(dict(zip(vocab_common, Wn)), open(str(output_dir)))
