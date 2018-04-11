# -*- coding: utf-8 -*-
"""
 Evaluation functions
"""
import logging
import numpy as np
#from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.datasets.base import Bunch
from .datasets.similarity import fetch_MEN, fetch_WS353, fetch_SimLex999, fetch_MTurk, fetch_RG65, fetch_RW
from .datasets.categorization import fetch_AP, fetch_battig, fetch_BLESS, fetch_ESSLI_1a, fetch_ESSLI_2b, \
    fetch_ESSLI_2c
from .analogy import *
from six import iteritems
from .embedding import Embedding

from .evaluate import *

logger = logging.getLogger(__name__)

def evaluate_on_semeval_2012_2_on_demand(w, words_demanded):
    """
    Simple method to score embedding using SimpleAnalogySolver

    Parameters
    ----------
    w : Embedding or dict
      Embedding or dict instance.

    Returns
    -------
    result: pandas.DataFrame
      Results with spearman correlation per broad category with special key "all" for summary
      spearman correlation
    """
    if isinstance(w, dict):
        w = Embedding.from_dict(w)

    data = fetch_semeval_2012_2()

    X, y = {}, {}
    for category, questions in data.X.iteritems():
        q, s = [], []
        for idx, question in enumerate(questions):
            for word in question:
                if word not in words_demanded:
                    break
                else:
                    pass
            else:
                q.append(question)
                s.append(data.y[category][idx])
        X[category] = np.array(q)
        vacancy = len(data.X)-len(X)
        logger.info('[WM]Failed to answer {} questions.'.format(vacancy))
        y[category] = np.array(s)

    data = Bunch(X_prot=data.X_prot, X=X, y=y,
                 categories_names=data.categories_names,
                 categories_descriptions=data.categories_descriptions)

    mean_vector = np.mean(w.vectors, axis=0, keepdims=True)
    categories = data.y.keys()
    results = defaultdict(list)
    for c in categories:
        # Get mean of left and right vector
        prototypes = data.X_prot[c]
        prot_left = np.mean(np.vstack(w.get(word, mean_vector) for word in prototypes[:, 0]), axis=0)
        prot_right = np.mean(np.vstack(w.get(word, mean_vector) for word in prototypes[:, 1]), axis=0)

        questions = data.X[c]
        question_left, question_right = np.vstack(w.get(word, mean_vector) for word in questions[:, 0]), \
                                        np.vstack(w.get(word, mean_vector) for word in questions[:, 1])

        scores = np.dot(prot_left - prot_right, (question_left - question_right).T)

        c_name = data.categories_names[c].split("_")[0]
        # NaN happens when there are only 0s, which might happen for very rare words or
        # very insufficient word vocabulary
        cor = scipy.stats.spearmanr(scores, data.y[c]).correlation
        results[c_name].append(0 if np.isnan(cor) else cor)

    final_results = OrderedDict()
    final_results['all'] = sum(sum(v) for v in results.values()) / len(categories)
    for k in results:
        final_results[k] = sum(results[k]) / len(results[k])
    return pd.Series(final_results)


def evaluate_words_demanded_on_all(w, cosine_similarity, words_demanded=None):
    if words_demanded is None:
        return evaluate_on_all(w, cosine_similarity)

    if isinstance(w, dict):
        w = Embedding.from_dict(w)

    # Calculate results on similarity
    logger.info("Calculating similarity benchmarks on words words_demanded")
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

    similarity_results = {}

    for name, data in iteritems(similarity_tasks):
        X, y = [], []
        for word_pair, score in zip(data.X, data.y):
            for word in word_pair:
                if word not in words_demanded:
                    break
                else:
                    pass
            else:
                X.append(word_pair)
                y.append(score)
        X = np.array(X)
        vacancy = len(data.X)-len(X)
        logger.info('[WM]Failed to answer {} questions.'.format(vacancy))
        y = np.array(y)
        similarity_results[name] = evaluate_similarity(w, X, y, cosine_similarity)
        logger.info("Spearman correlation of scores of {} out of {} word pairs of {}: {}".format(len(X), len(data.X), name, similarity_results[name]))

    # Calculate results on analogy
    logger.info("Calculating analogy benchmarks")
    analogy_tasks = {
        "Google": fetch_google_analogy(),
        "MSR": fetch_msr_analogy()
    }
    analogy_results = {}

    for name, data in iteritems(analogy_tasks):
        X, y = [], []
        for question, answer in zip(data.X, data.y):
            for word in question + answer:
                if word not in words_demanded:
                    break
                else:
                    pass
            else:
                X.append(question)
                y.append(answer)
        X = np.array(X)
        vacancy = len(data.X)-len(X)
        logger.info('[WM]Failed to answer {} questions.'.format(vacancy))
        y = np.array(y)
        analogy_results[name] = evaluate_analogy(w, X, y)
        logger.info("Analogy prediction accuracy on {} {}".format(name, analogy_results[name]))

    analogy_results["SemEval2012_2"] = evaluate_on_semeval_2012_2_on_demand(w, words_demanded)['all']
    logger.info("Analogy prediction accuracy on {} {}".format("SemEval2012", analogy_results["SemEval2012_2"]))

    # Calculate results on categorization
    logger.info("Calculating categorization benchmarks")
    categorization_tasks = {
        "AP": fetch_AP(),
        "BLESS": fetch_BLESS(),
        "Battig": fetch_battig(),
    }

    categorization_results = {}

    # Calculate results using helper function
    for name, data in iteritems(categorization_tasks):
        X = [word for word in data.X if word in words_demanded]
        y = [data.y[idx] for idx, word in enumerate(data.X) if word in words_demanded]
        X = np.array(X)
        vacancy = len(data.X)-len(X)
        logger.info('[WM]Failed to answer {} questions.'.format(vacancy))
        y = np.array(y)
        categorization_results[name] = evaluate_categorization(w, X, y)
        logger.info("Cluster purity on {} {}".format(name, categorization_results[name]))
    
    # Construct pd table
    cat = pd.DataFrame([categorization_results])
    analogy = pd.DataFrame([analogy_results])
    sim = pd.DataFrame([similarity_results])
    results = cat.join(sim).join(analogy)

    return results
