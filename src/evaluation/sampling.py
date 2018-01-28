from numpy import random

def reservoir(population, nsamples):
    samples = []
    for idx, item in enumerate(population):
        if len(samples)<nsamples:
            samples.append(item)
        elif int(random.uniform(0, idx)) < nsamples:
            samples[int(random.uniform(0, nsamples))] = item
    return samples


