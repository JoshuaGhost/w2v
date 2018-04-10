import unittest
from .utils import gensim2lexi_ordered, read_wv_csv, load_embeddings
from gensim.models import Word2Vec
from numpy.linalg import norm

base_dir = '/home/assassin/workspace/master_thesis/w2v/subs_test_extrinsic/'
files = [base_dir+'m1.csv', base_dir+'m2.csv']

class Gensim2LexiOrderedTestCase(unittest.TestCase):
    def setUp(self):
        self.model = Word2Vec(list(word.split() for word in 'a b c d e.b c d e.a b c c c a b b e a'.split('.')),
                              min_count=1, size=10)

    def test_gensim2lexi_ordered(self):
        vocab, vectors = gensim2lexi_ordered(self.model, True)
        self.assertEqual(vocab, [c for c in 'abcde'])
        self.assertEqual(vectors.shape, (5,10))
        self.assertAlmostEqual(norm(vectors[0,:]), norm(vectors[1,:]))
        self.assertAlmostEqual(norm(vectors[1,:]), 1.)


class ReadWvCsvTestCase(unittest.TestCase):
    def setUp(self):
        pass

    def test_read_wv_csv(self):
        vvs = list(map(read_wv_csv, files))
        vocab0, vecs0 = vvs[0]
        self.assertEqual(vocab0, [c for c in 'abcdefghijk'])
        self.assertEqual(vecs0.shape, (11,5))


class LoadEmbeddingsTestCase(unittest.TestCase):
    def setUp(self):
        pass

    def test_load_embeddings(self):
        vvs = load_embeddings(folder=base_dir, filename='m', extension='csv',
                              use_norm=True, input_format='csv', output_format='lovv')
        self.assertEqual(len(vvs), 2)
        vv0, vv1 = vvs
        self.assertEqual(vv0[0], [c for c in 'abcdefghijk'])
        self.assertEqual(vv0[1].shape, (11,5))


if __name__ == '__main__':
    unittest.main()
