from sampling import reservoir
from pathlib2 import Path
from numpy import log
import pickle
import logging
from utils import KL_divergence, hist2dist, smoothen, flatten

def wc(doc):
    counts = {}
    for line in doc:
        for word in line.split():
            stem = word.strip().lower()
            if stem in counts:
                counts[stem] += 1
            else:
                counts[stem] = 1
    return counts

def bic(doc, wlength, bigram_vocab):
    counts = {}
    for line in doc:
        words = line.split()
        for wordidx, word in enumerate(words):
            if word not in bigram_vocab:
                continue
            window = words[max(0, wordidx-wlength): wordidx]
            window += words[wordidx+1: wordidx+wlength+1]
            for context in window:
                if context not in bigram_vocab:
                    continue
                pair = word+' '+context
                if pair in counts:
                    counts[pair] += 1
                else:
                    counts[pair] = 1
    return counts
    counts.sort(key=lambda x: x[1], reverse=True)

FORMAT="%(asctime)-15s %(message)s"
logging.basicConfig(format=FORMAT)
logger = logging.getLogger('samplesize-kldiv-correct')
logger.setLevel(logging.INFO)

denominators = [100000, 50000, 30000, 10000, 5000, 3000, 1000, 500, 300, 100, 50, 30, 10]
nsentences = 4227933
fname_corpus = Path("../../original_corpus/article_4227933_lines.txt").resolve()
tmp_folder = Path('/tmp/zzj/')
bigram_sample_factor = 0.5

debug = False
if debug:
    denominators = [5]
    nsentences = 10
    fname_corpus = Path("../../original_corpus/test.txt").resolve()
    tmp_folder = Path('/tmp/zzjtest/')
    bigram_sample_factor = 0.75

wlength = 10
nsamples = 10

fname_wc_origin = (tmp_folder/Path('wc_origin.pkl')).resolve()
fname_bic_origin = (tmp_folder/Path('bic_prigin.pkl')).resolve()
fname_bigram_vocab = (tmp_folder/Path('bigram_vocab.pkl')).resolve()

if __name__ == '__main__':
    try:
        wc_origin = pickle.load(open(str(fname_wc_origin)))
    except IOError:
        wc_origin = wc(open(str(fname_corpus)))
        pickle.dump(wc_origin, open(str(fname_wc_origin), 'w+'))

    logging.info('wc_origin loaded, {} words'.format(str(len(wc_origin))))
    ordered_vocab, wc_origin = flatten(wc_origin)
    wc_dist_origin = hist2dist([float(c) for c in wc_origin])
    wc_log_dist_origin = [log(d) for d in wc_dist_origin]

    try:
        bigram_vocab = pickle.load(open(str(fname_bigram_vocab)))
    except IOError:
        bigram_vocab = set(reservoir(ordered_vocab, len(wc_origin)**bigram_sample_factor))
        pickle.dump(bigram_vocab, open(str(fname_bigram_vocab), 'w+'))

    try:
        bic_origin = pickle.load(open(str(fname_bic_origin)))
    except IOError:
        bic_origin = bic(open(str(fname_corpus)), wlength, bigram_vocab)
        pickle.dump(bic_origin, open(str(fname_bic_origin), 'w+'))

    logger.info('bic_origin loaded, {} bigram turples'.format(str(len(bic_origin))))
    ordered_bigram, bic_origin = flatten(bic_origin)
    bic_dist_origin = hist2dist([float(c) for c in bic_origin])

    sov_word = [0.]*len(denominators)
    sov_bigram = [0.]*len(denominators)

    for sample_idx in range(nsamples):
        logger.info('processing sample #{}...'.format(str(sample_idx)))
        for d_idx, d in enumerate(denominators):
            logger.info('current sample size: 1/{}th of original corpus in count of sentences'.format(str(d)))
            sample_size = int(nsentences / float(d))
            sample = reservoir(open(str(fname_corpus), 'r'), sample_size)
            
            wc_sample = wc(sample[:sample_size])
            wc_flat = flatten(wc_sample, ordered_vocab)
            wc_dist = hist2dist(wc_flat)
            sov_word[d_idx] += (sov(P=wc_dist, Q=wc_dist_origin))

            bic_sample = bic(sample[:sample_size], wlength, bigram_vocab)
            bic_flat = flatten(bic_sample, ordered_bigram)
            bic_dist = hist2dist(bic_flat)
            sov_bigram[d_idx] += (sov(P=bic_dist, Q=bic_dist_origin))

            logger.info('sample #{}, sample size: 1/{}:'.format(str(sample_idx), str(d)))
            logger.info('average sum-of-variance for word dist so-far: {}'.format(str(sov_word[d_idx]/(sample_idx+1))))
            logger.info('average sum-of-variance for bigram so-far: {}'.format(str(sov_bigram[d_idx]/(sample_idx+1))))

    avg_sov_word = [k/nsamples for k in sov_word]
    avg_sov_bigram = [k/nsamples for k in sov_bigram]
    print avg_sov_word
    print avg_sov_bigram

    with open('samplesize-avgsov_word.csv', 'w+') as fout:
        fout.write('sample_size, average_sum-of-variance_word\n')
        for d, k in zip(denominators, avg_sov_word):
            fout.write('1/'+str(d)+', '+str(k)+'\n')

    with open('samplesize-avgsov_bigram.csv', 'w+') as fout:
        fout.write('sample_size, average_sum-of-variance_word\n')
        for d, k in zip(denominators, avg_sov_bigram):
            fout.write('1/'+str(d)+', '+str(k)+'\n')
