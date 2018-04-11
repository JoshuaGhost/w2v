import logging
import sys
import os
import pickle
from .web.evaluate_on_demand import evaluate_words_demanded_on_all
from .web.datasets.categorization import fetch_AP, fetch_battig, fetch_BLESS
from .web.datasets.similarity import fetch_MEN, fetch_WS353, fetch_SimLex999, fetch_MTurk, fetch_RG65, fetch_RW
from .web.evaluate import evaluate_analogy, evaluate_categorization, evaluate_similarity, evaluate_on_semeval_2012_2
from scalable_learning.lovv.utils import lovv2web
from .web.analogy import *
from .web.embedding import Embedding
program = os.path.basename(sys.argv[0])
logger = logging.getLogger(program)
logging.basicConfig(format='%(asctime)s: %(levelname)s: %(message)s')
logging.root.setLevel(level=logging.INFO)


def eval_one_dataset(embeddings, merge=None, dataset='MEN'):
    if isinstance(embeddings, Embedding):
        web_combined = embeddings
    elif isinstance(embeddings[0], Embedding):
        web_combined = merge(embeddings)
    else:
        web_combined = lovv2web(embeddings)

    similarity_tasks = {
        "MEN": fetch_MEN(),
        "WS353": fetch_WS353(),
        "WS353R": fetch_WS353(which="relatedness"),
        "WS353S": fetch_WS353(which="similarity"),
        "SimLex999": fetch_SimLex999(),
        "RW": fetch_RW(),
        "RG65": fetch_RG65(),
        "MTurk": fetch_MTurk(),
    }

    analogy_tasks = {
        "Google": fetch_google_analogy(),
        "MSR": fetch_msr_analogy()
    }

    categorization_tasks = {
        "AP": fetch_AP(),
        "BLESS": fetch_BLESS(),
        "Battig": fetch_battig(),
        # "ESSLI_2c": fetch_ESSLI_2c(),
        # "ESSLI_2b": fetch_ESSLI_2b(),
        # "ESSLI_1a": fetch_ESSLI_1a()
    }

    if dataset in {'MEN', 'RW', 'WS353', 'WS353S', 'WS353R'}:
        logger.info("Calculating similarity benchmarks")
        data = similarity_tasks[dataset]
        result = evaluate_similarity(web_combined, data.X, data.y, cosine_similarity=False)
        logger.info("Spearman correlation of scores on {} {}".format(dataset, result))
    elif dataset in {'Google'}:
        logger.info("Calculating analogy benchmarks")
        data = analogy_tasks[dataset]
        result = evaluate_analogy(web_combined, data.X, data.y)
        logger.info("Analogy prediction accuracy on {} {}".format(dataset, result))
    elif dataset == 'SemEval2012':
        logger.info("Calculating analogy benchmarks")
        result = evaluate_on_semeval_2012_2(web_combined)['all']
        logger.info("Analogy prediction accuracy on {} {}".format("SemEval2012", result))
    else:
        logger.info("Calculating categorization benchmarks")
        data = categorization_tasks[dataset]
        result = evaluate_categorization(web_combined, data.X, data.y)
        logger.info("Cluster purity on {} {}".format(dataset, result))

    return result


if __name__=="__main__":

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

