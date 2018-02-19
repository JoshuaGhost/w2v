from mrjob.job import MRJob
from heapq import heappush, heapreplace
import random

class MRReservoirSampling(MRJob):

    def __init__(self, *args, **kwargs):
        super(MRReservoirSampling, self).__init__(*args, **kwargs)
        self.k = 100
    
    def mapper_init():
        self.H = [[] for i in range(self.k)]

    def mapper(self, _, line):
        r = random.random()
        for sample in self.H:
            if len(sample) < self.k:
                heappush(sample, (r, line))
            elif r > sample[0][0]:
                heapreplace(self.H, (r, line))

    def map_final(self):
        for sample_idx, sub_sample in enumerate(self.H):
            yield sample_idx, sub_sample

    def reducer_init(self):
        self.output = []

    def reducer(self, key, sub_sample):
        self.output += list(sub_sample)
        
    def reducer_final(self):
        ret = sort(sub_sample, key = sub_sampe[0])
        for idx, pair in enumerate(ret):
            if idx == self.k:
                break
            yield None, pair[1]

if __name__=='__main__':
    MRReservoirSampling.run()
