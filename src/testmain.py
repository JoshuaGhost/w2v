import unittest
import numpy as np
from main import make_index_table

class MakeIndexTabelTestCase(unittest.TestCase):
    def test_make_index_tabel(self):
        part = [c for c in 'ace']
        full = [c for c in 'abcdef']
        vec = np.asarray([[1,2,3],[2,3,4],[3,4,5],[4,5,6],[5,6,7],[6,7,8]])
        index_table = make_index_table(part, full)
        for a, b in zip(index_table, [0,2,4]):
           assert (a==b)
        for a, b in zip(vec[index_table], np.asarray([[1,2,3],[3,4,5],[5,6,7]])):
            for x, y in zip(a,b):
                assert(x==y)


if __name__ == '__main__':
    unittest.main()
