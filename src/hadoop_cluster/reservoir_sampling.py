import copy, random

def reservoir(source, nsamples):
    ret = []
    for idx, term in enumerate(source):
        if len(ret)<nsamples:
            ret.append(copy.deepcopy(term))
        else:
            if int(random.uniform(0, idx)) < nsamples:
                ret[int(random.uniform(0, nsamples))] = copy.deepcopy(term)
    return ret

if __name__=="__main__":
    a = "in the previous lectures we looked at second order linear homogeneous\
    equations with constant coefficients whose characteristic equation has\
    either different real roots or complex roots".split()
    print reservoir(a, 10)
