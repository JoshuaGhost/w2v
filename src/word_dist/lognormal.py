import numpy as np
from scipy.stats import lognorm
from numpy import pi, sqrt, log, exp
from collections import Counter

import matplotlib.pyplot as plt
import pylab as pl

'''
hist = [0.9926876983741216,
0.005424088658730165,
0.0010764445964539062,
0.0003387098796137099,
0.00017172023226225795,
9.177334726963649e-05,
5.865925289399445e-05,
3.311409437564203e-05,
2.6491275500513622e-05,
2.0104985870925518e-05,
1.6320517942280715e-05,
1.158993303147471e-05,
8.278523593910507e-06,
7.332406611749306e-06,
7.332406611749306e-06,
4.257526419725404e-06,
5.676701892967205e-06,
3.0748801920239026e-06,
4.4940556652657035e-06,
2.8383509464836024e-06]
'''

def KL_divergence(P, Q, loggedQ=False):
    if not loggedQ:
        Q = [log(q) if q!=0 else log(epsilon) for q in Q]
    return sum([p*(log(p)-q) if p > 0. else 0. for p,q in zip(P,Q)])

def lognorm_para_estimate(hist, include_zero=True, epsilon=10**(-13)):
    if not include_zero:
        offset = 1
    else:
        offset = 0
    sum_hist = sum(hist[offset:])
    mu = sum(log(c+offset)*h if (c+offset)!=0 else log(epsilon) * h for c, h in enumerate(hist[offset:]))/sum_hist
    ssquare = sum(h*(log(c+offset))**2 if (c+offset)!=0 else h*(log(epsilon))**2 for c, h in enumerate(hist[offset:]))/sum_hist-mu**2
    #ssquare = sum(h*(log(c+offset)-mu)**2 if (c+offset)!=0 else h*(log(epsilon)-mu)**2 for c, h in enumerate(hist[offset:]))/sum(h*(c+offset)*sqrt(2*pi) for c, h in enumerate(hist[offset:]))
    s = sqrt(ssquare)
    return mu, s

def smoothen(d, smoothen_factor):
    noz = Counter(d)[0]
    return [(1-smoothen_factor*noz)*p if p!=0 else smoothen_factor for p in d]

if __name__=='__main__':
    hist_threshold = 60
    scale = 15
    hist = [0] * hist_threshold
    for line in open('workspace/master_thesis/corpora/test.txt'):
        wc = Counter(line.split())
        count = wc['love']
        if count<60:
            hist[count]+=1

    hist = smoothen(hist, 1)

    sum_hist = sum(hist)
    dist = [h/float(sum_hist) for h in hist]

    epsilon = 0.5
    for scale in range(10, 21):
        epsilon = 10.**(-6)
        mu, s = lognorm_para_estimate(hist, True, epsilon)
        p = lognorm.pdf([i for i in range(hist_threshold)], scale=exp(mu), s=s)

        print KL_divergence(dist[1:], [i*scale for i in p[1:]])
        print dist
        print p

        pl.figure(figsize=(10,10))
        plt.plot([log(i*scale) for i in p[1:]], [log(i) for i in dist[1:]], 'ro')
        plt.plot([i for i in range(-8, 0)], [i for i in range(-8, 0)], 'b-')
        #plt.show()
        pl.savefig('pics/scale{}.png'.format(str(scale)))
        plt.close()
