from sampling import reservoir
from pathlib2 import Path
from numpy import log
import pickle
import logging

FORMAT="%(asctime)-15s %(message)s"
logging.basicConfig(format=FORMAT)
logger = logging.getLogger('samplesize-kldiv')
logger.setLevel(logging.INFO)

denominator_power_max = 5#1/100000 th of origin, 42 sentences per sample
denominator_power_min = 1#1/10 th of origin, 422793 sentences per sample
nsamples = 40
nsentences = 4227933
wlength = 10

fname_corpus = Path("../../original_corpus/article_4227933_lines.txt").resolve()
tmp_folder = Path('/tmp/zzj/')
fname_wc_origin = (tmp_folder/Path('wc_origin.pkl')).resolve()
fname_bic_origin = (tmp_folder/Path('bic_prigin.pkl')).resolve()
fname_bigram_vocab = (tmp_folder/Path('bigram_vocab.pkl')).resolve()
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

def flatten(counts_dict, ordered_keys=None):
    if ordered_keys is None:
        return [[key, [counts_dict[key]]] for key in counts_dict]
    return [counts_dict[key] if key in counts_dict else 0 for key in ordered_keys]
                
def KL_divergence(P, Q, loggedQ=False):
    if not loggedQ:
        Q = [log(q) for q in Q]
    return sum([x[0]*(log(x[0])-x[1]) if x[0] > 0. else 0. for x in zip(P,Q)])

if __name__ == '__main__':
    try:
        wc_origin = pickle.load(open(str(fname_wc_origin)))
    except IOError:
        wc_origin = wc(open(str(fname_corpus)))
        pickle.dump(wc_origin, open(str(fname_wc_origin), 'w+'))

    wc_origin = [[key, wc_origin[key]] for key in wc_origin]
    ordered_vocab = [c[0] for c in wc_origin]
    word_count_origin = [c[1] for c in wc_origin]
    word_total_origin = sum(word_count_origin)
    word_log_origin = [log(c/float(word_total_origin)) for c in word_count_origin]
    logging.info('wc_origin loading, {} words, total count {}'.format(str(len(ordered_vocab)), str(int(word_total_origin))))

    try:
        bigram_vocab = pickle.load(open(str(fname_bigram_vocab)))
    except IOError:
        bigram_vocab = set(reservoir(ordered_vocab, int(word_total_origin**0.5)))
        pickle.dump(bigram_vocab, open(str(fname_bigram_vocab), 'w+'))
    try:
        bic_origin = pickle.load(open(str(fname_bic_origin)))
    except IOError:
        bic_origin = bic(open(str(fname_corpus)), wlength, bigram_vocab)
        pickle.dump(bic_origin, open(str(fname_bic_origin), 'w+'))

    bic_origin = [[key, bic_origin[key]] for key in bic_origin]
    ordered_bigram = [c[0] for c in bic_origin]
    bigram_count_origin = [c[1]for c in bic_origin]
    bigram_total_origin = word_total_origin * wlength * 2 - (1 + wlength) * wlength * nsentences
    bigram_log_origin = [log(c/float(bigram_total_origin)) for c in bigram_count_origin]
    logger.info('bic_origin loaded, {} bigram turples, total count {}'.format(str(len(bic_origin)), str(int(bigram_total_origin))))

    kld_word = []
    kld_bigram = []

    word_count_accus = []
    bigram_count_accus = []
    word_total_accus = []
    bigram_total_accus = []

    for sample_idx in range(nsamples):
        denominator = 10
        sample_size = int(nsentences / float(denominator))
        sample = reservoir(open(str(fname_corpus), 'r'), sample_size)
        logger.info('processing sample #{}...'.format(str(sample_idx)))
        for denominator_power in range(denominator_power_min, denominator_power_max+1):
            denominator = 10**denominator_power
            accu_idx = denominator_power - 1
            sample_size = int(nsentences / float(denominator))
            logger.info('counting terms from sample #{}, sample size: {}, denominator: {}'.format(str(sample_idx), str(sample_size), str(denominator)))

            wc_sample = wc(sample[:sample_size])
            wc_flat = flatten(wc_sample, ordered_vocab)
            try:
                word_count_accus[accu_idx] = [x[0]+x[1] for x in zip(word_count_accus[accu_idx], wc_flat)]
            except IndexError:
                word_count_accus.append(wc_flat)

            bic_sample = bic(sample[:sample_size], wlength, bigram_vocab)
            bic_flat = flatten(bic_sample, ordered_bigram)
            try:
                bigram_count_accus[accu_idx] = [x[0]+x[1] for x in zip(bigram_count_accus[accu_idx], bic_flat)]
            except IndexError:
                bigram_count_accus.append(bic_flat)

    for denominator_power in range(denominator_power_min, denominator_power_max+1):
        denominator = 10**denominator_power
        idx = denominator_power - 1#accu[0]s are for 1/10 th
        word_total_accu = sum(word_count_accus[idx])
        word_prop_accu = [float(c)/word_total_accu for c in word_count_accus[idx]]
        kld_word.append(KL_divergence(P=word_prop_accu, Q=word_log_origin, loggedQ=True))
        logger.info('sample size: 1/{}'.format(str(denominator)))
        logger.info('KL divergence for word: {}'.format(str(kld_word[-1])))

        bigram_total_accu = sum(bigram_count_accus[idx])
        bigram_prop_accu = [float(c)/bigram_total_accu for c in bigram_count_accus[idx]]
        kld_bigram.append(KL_divergence(P=bigram_prop_accu, Q=bigram_log_origin, loggedQ=True))
        logger.info('KL divergence for bigram: {}'.format(str(kld_bigram[-1])))

    print kld_word
    print kld_bigram
    with open('samplesize-kldiv_word.csv', 'w+') as fout:
        for denominator_power in range(denominator_power_min, denominator_power_max+1):
            denominator = 10**denominator_power
            fout.write("1/"+str(denominator)+', ')
        fout.write('\n')
        for kld in kld_word:
            fout.write(str(kld)+', ')
        fout.write('\n')

    with open('samplesize-kldiv_bigram.csv', 'w+') as fout:
        for denominator_power in range(denominator_power_min, denominator_power_max+1):
            denominator = 10**denominator_power
            fout.write("1/"+str(denominator)+', ')
        fout.write('\n')
        for kld in kld_bigram:
            fout.write(str(kld)+', ')
        fout.write('\n')

