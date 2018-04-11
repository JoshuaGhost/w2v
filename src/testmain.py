import unittest

import numpy as np

from main import load_or_train
from main import make_index_table

base_dir = '/home/assassin/workspace/master_thesis/w2v/'
dumps = [base_dir + 'subs_test_extrinsic/m1.csv', base_dir + 'subs_test_extrinsic/m2.csv']
corpora = [base_dir + 'test_subcopora/c0', base_dir + 'test_subcopora/c1']


class MakeIndexTabelTestCase(unittest.TestCase):
    def test_make_index_tabel(self):
        part = [c for c in 'ace']
        full = [c for c in 'abcdef']
        vec = np.asarray([[1, 2, 3], [2, 3, 4], [3, 4, 5], [4, 5, 6], [5, 6, 7], [6, 7, 8]])
        index_table = make_index_table(part, full)
        for a, b in zip(index_table, [0, 2, 4]):
            assert (a == b)
        for a, b in zip(vec[index_table], np.asarray([[1, 2, 3], [3, 4, 5], [5, 6, 7]])):
            for x, y in zip(a, b):
                assert (x == y)


class LoadOrTrainCase(unittest.TestCase):
    def setUp(self):
        pass

    def test_load_or_train(self):
        m = load_or_train(corpora[0], size=5, filter_vocab=['the'], negative=5, workers=10, min_count=1, use_norm=True,
                          dump_format='csv', dump_file='/tmp/test.csv', output_format='lovv')
        assert ('the' not in m[0])
        assert (len(m[0]) > 0)
        m_prime = load_or_train(corpora[0], size=5, filter_vocab=['the'], negative=5, workers=10, min_count=1,
                                use_norm=True, dump_format='csv', dump_file='/tmp/test.csv', output_format='lovv')
        self.assertEqual(m[0], m_prime[0])
        self.assertEqual(m[1].all(), m_prime[1].all())


class MakeMaskTestCase(unittest.TestCase):
    def test_make_mask(self):
        part = [c for c in 'aceghi']
        vocab = [c for c in 'abcdefghij']
        mask = make_mask(part, vocab)
        for a, b in zip(mask, np.asarray([True, False, True, False, True, False, True, True, True, False])):
            assert (a == b)


if __name__ == '__main__':
    unittest.main()
