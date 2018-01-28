from numpy import log
from collections import Counter

epsilon = 10**(-16)

def smoothen(d, smoothen_factor):
    sum_d = sum(d)
    noz = Counter(d)[0]
    return [p-smoothen_factor*noz*p/sum_d if p!=0 else smoothen_factor for p in d]

def KL_divergence(P, Q):
    logP = [log(p) for p in smoothen(P, epsilon)]
    logQ = [log(q) for q in smoothen(Q, epsilon)]
    return sum([p * (logp - logq) for p, (logp, logq) in zip(P, zip(logP, logQ))])

def hist2dist(h):
    s = sum(h)
    return [float(c)/s for c in h]

def flatten(counts_dict, ordered_keys=None):
    if ordered_keys is None:
        keys = [key for key in counts_dict]
        return keys, [counts_dict[key] for key in keys]
    return [counts_dict[key] if key in counts_dict else 0 for key in ordered_keys]

def sov(P, Q):
    return sum((p-q)**2 for p, q in zip(P, Q))

