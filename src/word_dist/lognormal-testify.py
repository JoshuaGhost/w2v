from sampling import reservoir
from pathlib2 import Path
from numpy import log, sqrt, exp
from utils import KL_divergence, smoothen, hist2dist
from scipy.stats import norm

from lognormal import KL_divergence, lognorm_para_estimate, smoothen
import pickle
import logging

FORMAT="%(asctime)-15s %(message)s"
logging.basicConfig(format=FORMAT)
logger = logging.getLogger('lognormal-testify')
logger.setLevel(logging.INFO)

epsilon = 10.**(-6)

nsentences = 4227933
fname_corpus = Path("../../original_corpus/article_4227933_lines.txt").resolve()
dumped_hist = Path('hist.pkl').resolve()
max_word_freq = 1000
wlength = 10

target_words = ['the', 'love', 'network']
target_bigrams = ['this is', 'man woman', 'soccer football', 'germany berlin', 'good better','tree oak']

debug = False 
if debug:
    nsentences = 100
    fname_corpus = Path("../../original_corpus/test.txt").resolve()
    dumped_hist = Path('test_hist.pkl').resolve()
    max_word_freq = 50
    wlength = 3

smoothen_factor = 0.001
           
def hist2dist(h):
    s = sum(h)
    return [float(c)/s for c in h]

def smoothen(d, smoothen_factor):
    noz = Counter(d)[0]
    return [(1-smoothen_factor*noz)*p if p!=0 else smoothen_factor for p in d]

if __name__ == '__main__':
    try:
        hist = pickle.load(open(str(dumped_hist)))
        logger.info('histogram cache loaded')
    except IOError:
        logger.info('histogram cache unfound, building cache...')
        hist = dict([(t, [0 for i in range(max_word_freq)]) for t in target_words+target_bigrams])
        for idx, line in enumerate(open(str(fname_corpus))):
            words = line.split()
            wc = Counter(words)
            for word in target_words:
                if wc[word]<max_word_freq:
                    hist[word][wc[word]] += 1

            bigrams = []
            for offset in range(1, wlength+1):
                bigrams += [w+' '+c for w, c in zip(words[:-offset], words[offset:])]
                bigrams += [c+' '+w for w, c in zip(words[:-offset], words[offset:])]
            bic = Counter(bigrams)
            for bigram in target_bigrams:
                if wc[bigram]<max_word_freq:
                    hist[bigram][wc[bigram]] +=1

            if (idx + 1) % 100 == 0:
                logger.info('{} line processed'.format(idx+1))

        pickle.dump(hist, open(str(dumped_hist), 'w+'))
        logger.info('histogram dumped')
