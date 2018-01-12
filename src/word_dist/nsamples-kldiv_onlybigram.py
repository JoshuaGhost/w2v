from sampling import reservoir
from pathlib2 import Path
from numpy import log
import pickle
import logging

FORMAT="%(asctime)-15s %(message)s"
logging.basicConfig(format=FORMAT)
logger = logging.getLogger('nsamples-kldiv')
logger.setLevel(logging.INFO)

sample_size = 42279
nsample_max = 500
nsample_step = 10
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
    #word_log_origin = [log(c/float(word_total_origin)) for c in word_count_origin]
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
    bigram_total_origin = sum(bigram_count_origin)
    bigram_log_origin = [log(c/float(bigram_total_origin)) for c in bigram_count_origin]
    logger.info('bic_origin loaded, {} bigram turples, total count {}'.format(str(len(bic_origin)), str(int(bigram_total_origin))))

    wc_accu = {}
    bic_accu = {}
    
    nsamples = [i for i in range(nsample_step, nsample_max, nsample_step)]
    #kld_word = []
    kld_bigram = []

    for nsample in range(nsample_max):
        logger.info('sample #{} under processing...'.format(str(nsample)))
        sample = reservoir(open(str(fname_corpus), 'r'), sample_size)
        #wc_sample = wc(sample)
        bic_sample = bic(sample, wlength, bigram_vocab)
        #for word in wc_sample:
        #    if word in wc_accu:
        #        wc_accu[word] += wc_sample[word]
        #    else:
        #        wc_accu[word] = wc_sample[word]
        for bigram in bic_sample:
            if bigram in bic_accu:
                bic_accu[bigram] += bic_sample[bigram]
            else:
                bic_accu[bigram] = bic_sample[bigram]
        if (nsample+1) % nsample_step == 0:
            #word_count_accu = flatten(wc_accu, ordered_vocab)
            #word_total_accu = sum(word_count_accu)
            #word_prop_accu = [float(c)/word_total_accu for c in word_count_accu]
            #kld_word.append(KL_divergence(P=word_prop_accu, Q=word_log_origin, loggedQ=True))
            logger.info("sample count: {}".format(str(nsample)))
            #logger.info("KL divergence for word: {}".format(str(kld_word[-1])))
            
            bigram_count_accu = flatten(bic_accu, ordered_bigram)
            bigram_total_accu = sum(bigram_count_accu)
            bigram_prop_accu = [float(c)/bigram_total_accu for c in bigram_count_accu]
            kld_bigram.append(KL_divergence(P=bigram_prop_accu, Q=bigram_log_origin, loggedQ=True))
            logger.info('KL divergence for bigram: {}'.format(str(kld_bigram[-1])))


    #print kld_word
    print kld_bigram
    #with open('nsamples-kldiv_word.csv', 'a+') as fout:
    #    for sample_count in nsamples:
    #        fout.write(str(sample_count)+', ')
    #    fout.write('\n')
    #    for kld in kld_word:
    #        fout.write(str(kld)+', ')
    with open('nsamples-kldiv_bigram.csv', 'a+') as fout:
        for sample_count in nsamples:
            fout.write(str(sample_count)+', ')
        fout.write('\n')
        for kld in kld_bigram:
            fout.write(str(kld)+', ')
        fout.write('\n')

