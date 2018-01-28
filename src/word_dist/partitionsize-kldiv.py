from sampling import reservoir
from pathlib2 import Path
from numpy import log
import pickle
import logging
import pandas as pd
from multiprocessing import Pool

FORMAT="%(asctime)-15s %(message)s"
logging.basicConfig(format=FORMAT)
logger = logging.getLogger('partitionsize-kld')
logger.setLevel(logging.INFO)

denominators = [100000, 50000, 30000, 10000, 5000, 3000, 1000, 500, 300, 100, 50, 30, 10]
nsentences = 4227933
fname_corpus = Path("../../original_corpus/article_4227933_lines.txt").resolve()
tmp_folder = Path('/tmp/zzj/')
bigram_sample_factor = 0.5
word_oname = 'part-kld_word.csv'
bigram_oname = 'part-kld_bigram.csv'
debug = True
if debug:
    denominators = [5, 10, 100]
    nsentences = 10000
    fname_corpus = Path("../../original_corpus/test.txt").resolve()
    tmp_folder = Path('/tmp/zzjtest/')
    bigram_sample_factor = 0.75
    word_oname = 'debug_' + word_oname
    bigram_oname = 'debug_' + bigram_oname

wlength = 10
nsamples = 10

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
        keys = [key for key in counts_dict]
        return keys, [counts_dict[key] for key in keys]
    return [counts_dict[key] if key in counts_dict else 0 for key in ordered_keys]
                
def KL_divergence(P, Q, loggedQ=False):
    if not loggedQ:
        Q = [log(q) for q in Q]
    return sum([p*(log(p)-q) if p > 0. else 0. for p,q in zip(P,Q)])

def hist2dist(h):
    s = sum(h)
    if s == 0:
        return [0. for c in h]
    return [float(c)/s for c in h]

def divergent(count, ordered_keys, Q, denominator):
    flat_count = flatten(count, ordered_keys)
    dist = hist2dist(flat_count)
    kld_current = KL_divergence(P=dist, Q=Q, loggedQ=True)
    return pd.DataFrame([['1/'+str(denominator), kld_current]], columns=['partition_size', 'KL_Divergence'])

def worker(argvs):
    d_idx = argvs[0]
    d = argvs[1]
    df_word = pd.DataFrame(columns=['partition_size', 'KL_Divergence_word'])
    df_bigram = pd.DataFrame(columns=['partition_size', 'KL_Divergence_bigram'])
    partition_size = int(nsentences/float(d))+1
    logger.info('partition size: 1/{}th of origin'.format(d))
    partition = []
    partition_idx = 0
    for line_idx, line in enumerate(open(str(fname_corpus), 'r')):
        partition.append(line)
        if (line_idx+1) % partition_size == 0:
            logger.info('partition #{} of size 1/{}th'.format(partition_idx, d))
            wc_partition = wc(partition)
            logger.info('partition #{} of size 1/{}th, word count complete'.format(partition_idx, d))
            df_word_current = divergent(count=wc_partition, ordered_keys=ordered_vocab, Q=wc_log_dist_origin, denominator=d)
            logger.info('partition #{} of size 1/{}th, df_word_current calculation complete'.format(partition_idx, d))
            df_word = df_word.append(df_word_current)

            bic_partition = bic(partition, wlength, bigram_vocab)
            logger.info('partition #{} of size 1/{}th, bigram count complete'.format(partition_idx, d))
            df_bigram_current = divergent(count=bic_partition, ordered_keys=ordered_bigram, Q=bic_log_dist_origin, denominator=d)
            logger.info('partition #{} of size 1/{}th, df_bigram_current calculation complete'.format(partition_idx, d))
            df_bigram = df_bigram.append(df_bigram_current)
            
            logger.info('partition index: {}, sample size: 1/{}th finished'.format(str(partition_idx), str(d)))
            partition_idx += 1
            partition = []
            if partition_idx == 10:
                break
        else:
            pass
    else:
        if (line_idx+1) % partition_size != 0:
            wc_partition = wc(partition)
            df_word = df_word.append(divergent(count=wc_partition, ordered_keys=ordered_vocab, Q=wc_log_dist_origin, denominator=d))

            bic_partition = bic(partition, wlength, bigram_vocab)
            df_bigram = df_bigram.append(divergent(count=bic_partition, ordered_keys=ordered_bigram, Q=bic_log_dist_origin, denominator=d))
            
            logger.info('partition index: {}, sample size: 1/{}th finished'.format(str(partition_idx), str(d)))
    return df_word, df_bigram

if __name__ == '__main__':
    try:
        wc_origin = pickle.load(open(str(fname_wc_origin)))
    except IOError:
        wc_origin = wc(open(str(fname_corpus)))
        pickle.dump(wc_origin, open(str(fname_wc_origin), 'w+'))

    logger.info('wc_origin loaded, {} words'.format(str(len(wc_origin))))
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
    bic_log_dist_origin = [log(d) for d in bic_dist_origin]
    
    '''
    df_word = pd.DataFrame(columns=['partition_size', 'KL_Divergence'])
    df_bigram = pd.DataFrame(columns=['partition_size', 'KL_Divergence'])
    for d_idx, d in enumerate(denominators):
        partition_size = int(nsentences/float(d))+1
        logger.info('partition size: 1/{}th of origin'.format(d))
        partition = []
        partition_idx = 0
        print partition_size, nsentences, d
        for line_idx, line in enumerate(open(str(fname_corpus), 'r')):
            partition.append(line)
            if (line_idx+1) % partition_size == 0:
                wc_partition = wc(partition)
                df_word = df_word.append(divergent(count=wc_partition, ordered_keys=ordered_vocab, Q=wc_log_dist_origin, denominator=d))

                bic_partition = bic(partition, wlength, bigram_vocab)
                df_bigram = df_bigram.append(divergent(count=bic_partition, ordered_keys=ordered_bigram, Q=bic_log_dist_origin, denominator=d))
                
                logger.info('partition index: {}, sample size: 1/{}th finished'.format(str(partition_idx), str(d)))
                partition_idx += 1
                partition = []
            else:
                pass
        else:
            if (line_idx+1) % partition_size != 0:
                wc_partition = wc(partition)
                df_word = df_word.append(divergent(count=wc_partition, ordered_keys=ordered_vocab, Q=wc_log_dist_origin, denominator=d))

                bic_partition = bic(partition, wlength, bigram_vocab)
                df_bigram = df_bigram.append(divergent(count=bic_partition, ordered_keys=ordered_bigram, Q=bic_log_dist_origin, denominator=d))
                
                logger.info('partition index: {}, sample size: 1/{}th finished'.format(str(partition_idx), str(d)))
    '''
    pool = Pool(32)
    dfs = pool.map(worker, enumerate(denominators))
    #dfs = map(worker, enumerate(denominators))
    df_word = pd.concat([d[0] for d in dfs])
    df_bigram = pd.concat([d[1] for d in dfs])

    df_word.to_csv(word_oname)
    df_bigram.to_csv(bigram_oname)
