from gensim.models.word2vec import Word2Vec
from pathlib2 import Path
from numpy import exp
import logging

tmp_dir = Path('/tmp/zzj').resolve()
Ws_dump_dir = tmp_dir/Path('Ws.pkl')
Cs_dump_dir = tmp_dir/Path('Cs.pkl')
counts_dump_dir = tmp_dir/Path('counts.pkl')

FORMAT = "(asctime)-15s %(message)s"
logging.basicConfig(format=FORMAT)
logger = logging.getLogger('combining_document_new')
subs_dir = Path('../../models/subs').reslove()

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

    for idx, model_name in enumerate(subs_dir.glob("sub.*")):
        m = Word2Vec.load(str(model_name))
        wc = dict([[word, m.wv.vocab[word].count] for word in m.wv.vocab.keys()])
        vocab_size = len(wv.index2word)
        vocab = []
        count = []
        W = []
        C = []
        word2index = {}
        for word_index in xrange(vocab_size):
            word = m.wv.index2word[word_index]
            vocab.append(word)
            count.append(m.wv.vocab[word].count)
            W.append(m.wv.syn0[word])
            C.append(m.wv.syn1neg[word])
            word2index[word] = word_index
        vocabs.append(vocab)
        counts.append(count)
        Ws.append(W)
        Cs.append(C)
        word2indexes.append(word2index)
        '''
        !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        '''
        if idx>1:
            break

    vocab_common = list(reduce(lambda x,y: x.intersection(y), [set(vocab) for vocab in vocabs]))
    #word2index_common = dict([[word, idx] for idx, word in enumerate(vocab_common)])
    logger.info("sub-models loaded, {} words in common vocabulary".format(len(vocab_common)))

    for model_idx in xrange(num_models)
        W = []
        count = []
        C = []
        for word_new_idx, word in enumerate(vocab_common):
            word_idx = word2index[model_idx][word]
            W.append(Ws[model_idx][word_idx])
            C.append(Cs[model_idx][word_idx])
            count.append(counts[model_idx][word_idx])
        Ws[model_idx] = W
        Cs[model_idx] = C
        counts[model_idx] = count
    dump(Ws, open(str(Ws_dump_dir), 'w+'))
    dump(Cs, open(str(Cs_dump_dir), 'w+'))
    dump(counts, open(str(counts_dump_dir), 'w+'))
    dump(vocab_common, open(str(vocab_common_dump_dir), 'w+'))

num_models = len(Ws)
num_vocab = len(vocab_common)

logger.info("word-wise aligned, start to merging")

iteration = 10
Wn = np.random.random(())
for i in xrange(iteration):
    for idx, word in enumerate(vocab_common):
        for idx, context in enumerate(vocab_common):
            if word != context:
                goal = 
