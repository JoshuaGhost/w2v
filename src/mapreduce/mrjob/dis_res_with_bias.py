from mrjob.job import MRJob
import random

class Sampling(MRJob):
    def mapper(self, _, line):
        yield random.randint(1, 10), line

    def reducer(self, key, values):
        yield values


if __name__ == '__main__':
    Sampling.run()
