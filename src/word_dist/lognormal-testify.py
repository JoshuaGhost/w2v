from sampling import reservoir
from pathlib2 import Path
from numpy import log, sqrt, exp
from collections import Counter
from scipy.stats import norm
import pickle
import logging

FORMAT="%(asctime)-15s %(message)s"
logging.basicConfig(format=FORMAT)
logger = logging.getLogger('lognormal-testify')
logger.setLevel(logging.INFO)

nsentences = 4227933
fname_corpus = Path("../../original_corpus/article_4227933_lines.txt").resolve()
tmp_folder = Path('/tmp/zzj/')
bigram_sample_factor = 0.5
vocab_count = 10000
max_word_freq = 10000

debug = False
if debug:
    nsentences = 100
    fname_corpus = Path("../../original_corpus/test.txt").resolve()
    tmp_folder = Path('/tmp/zzjtest/')
    bigram_sample_factor = 0.75
    vocab_count = 100
    max_word_freq = 50

fname_vocab = (tmp_folder/Path('vocab.pkl')).resolve()
wlength = 10
epsilon = 0.01
smoothen_factor = 0.001
           
def KL_divergence(P, Q, loggedQ=False):
    if not loggedQ:
        Q = [log(q) for q in Q]
    return sum([p*(log(p)-q) if p > 0. else 0. for p,q in zip(P,Q)])

def hist2dist(h):
    s = sum(h)
    return [float(c)/s for c in h]

def smoothen(d, smoothen_factor):
    noz = Counter(d)[0]
    return [(1-smoothen_factor*noz)*p if p!=0 else smoothen_factor for p in d]

if __name__ == '__main__':
    try:
        vocab = pickle.load(open(str(fname_vocab)))
        logging.info('vocab file exists, use the dumped one')
    except IOError:
        logging.info('vocab file not exists, building vocab file')
        vocab = set()
        for line in open(str(fname_corpus)):
            for word in line.split():
               vocab = vocab.union({word})
        vocab = reservoir(list(vocab), vocab_count)
        pickle.dump(vocab, open(str(fname_vocab), 'w+'))
        logging.info('vocab file built, in total {} vocab sampled'.format(str(len(vocab))))

    hist = dict([(w, [0 for i in range(max_word_freq)]) for w in vocab])
    for line in open(str(fname_corpus)):
        wc = Counter(line.split())
        vtemp = set(wc.keys()).intersection(vocab)
        for word in vtemp:
            try:
                hist[word][wc[word]] += 1
            except IndexError:
                pass

    kld_avg = 0.0
    fout = open('lognormal_testify.csv','w+')
    fout.write('word, count, mu, sigma, KL_divergence\n')
    X = [log(y+0.5) for y in range(max_word_freq)]#X~N(mu, sigma)
    for w, h in hist.iteritems():
        logging.info('inspecting word {}'.format(w))
        sum_count = sum([count*nsen for count, nsen in enumerate(h)])
        #h = [(1-epsilon/sum_count)*c if c!=0 else epsilon for c in h]#laplacian smoothing
        dist = hist2dist(h)
        dist = smoothen(dist, smoothen_factor)
        mu = sum([log(c)*p if c!=0 else log(epsilon)*p for c, p in enumerate(dist)])
        sigma = sqrt(sum([p*(x-mu)**2 for x, p in zip(X, dist)]))
        n = list(norm.cdf(X, loc=mu, scale=sigma))
        n = [b-a for a, b in zip(n, n[1:]+[1,])]
        kld = KL_divergence(P=dist, Q=n)
        logging.info('P:=word frequence of \'{}\', Q:=lognormal(mu={}, sigma={}), KL(P||Q) = {}'.format(w, str(mu), str(sigma), str(kld)))
        fout.write(','.join((w, str(sum_count), str(mu), str(sigma), str(kld)))+'\n')
        kld_avg += kld

    logging.info('average KL divergence: '+str(kld_avg))
    fout.write(str(kld_avg/len(vocab)))
    fout.close()
