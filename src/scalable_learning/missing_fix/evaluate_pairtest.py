import unittest
from .evaluate_pair import common_words, intrinsic_cv_split, extrinsic_cv_split
import numpy as np


class CommonWordsTestCase(unittest.TestCase):
    def setUp(self):
        self.source = ([c for c in 'abcde'], [[1,2,3],[2,3,4],[3,4,5],[4,5,6],[5,6,7]])
        self.target = ([c for c in 'bdefg'], [[1,2,3],[2,3,4],[3,4,5],[4,5,6],[5,6,7]])
        self.source = ([c for c in 'abcdefijwx'],
                       [[1,2,3],[2,3,4],[3,4,5],[4,5,6],[5,6,7],[1,2,3],[2,3,4],[3,4,5],[4,5,6],[5,6,7]])
        self.target = ([c for c in 'bdefgiklmn'],
                       [[2,3,4],[3,4,5],[4,5,6],[5,6,7],[1,2,3],[2,3,4],[3,4,5],[4,5,6],[5,6,7],[1,2,3]])

    def test_common_words(self):
        vocab = []
        vectors_source = []
        vectors_target = []
        for word, vs, vt in common_words(self.source, self.target):
            vocab.append(word)
            vectors_source.append(vs)
            vectors_target.append(vt)
        self.assertEqual(vocab, [c for c in 'bdefi'])
        self.assertEqual(vectors_source, [[2, 3, 4], [4, 5, 6], [5, 6, 7], [1, 2, 3], [2, 3, 4]])
        self.assertEqual(vectors_target, [[2, 3, 4], [3, 4, 5], [4, 5, 6], [5, 6, 7], [2, 3, 4]])


class IntrinsicCVSplitTestCase(unittest.TestCase):
    def setUp(self):
        self.source = ([c for c in 'abcdefijwx'],
                       [[1,2,3],[2,3,4],[3,4,5],[4,5,6],[5,6,7],[1,2,3],[2,3,4],[3,4,5],[4,5,6],[5,6,7]])
        self.target = ([c for c in 'bdefgiklmn'],
                       [[2,3,4],[3,4,5],[4,5,6],[5,6,7],[1,2,3],[2,3,4],[3,4,5],[4,5,6],[5,6,7],[1,2,3]])

    def test_intrinsic_cv_split(self):
        trains, tests, validations = intrinsic_cv_split((self.source, self.target), sorted_vocab=True)
        self.assertEqual(trains[0][0], [c for c in 'bde'])
        self.assertEqual(trains[0][0], trains[1][0])
        self.assertEqual(tests[0][0], [c for c in 'fi'])
        self.assertEqual(tests[1][0], tests[0][0])
        self.assertEqual(validations[0][0], [])
        self.assertEqual(validations[0][0], validations[1][0])
        self.assertEqual(trains[0][1], [[2,3,4],[4,5,6],[5,6,7]])
        self.assertEqual(trains[1][1], [[2,3,4],[3,4,5],[4,5,6]])
        self.assertEqual(tests[0][1], [[1,2,3],[2,3,4]])
        self.assertEqual(tests[1][1], [[5,6,7],[2,3,4]])
        self.assertEqual(validations[0][1], [])
        self.assertEqual(validations[0][1], validations[1][1])


class ExtrinsicCVSplitTestCase(unittest.TestCase):
    def setUp(self):
        self.source = ([c for c in 'abcdefijwx'],
                       np.asarray([[1,2,3],[2,3,4],[3,4,5],[4,5,6],[5,6,7],[1,2,3],[2,3,4],[3,4,5],[4,5,6],[5,6,7]]))
        self.target = ([c for c in 'bdefgiklmn'],
                       np.asarray([[2,3,4],[3,4,5],[4,5,6],[5,6,7],[1,2,3],[2,3,4],[3,4,5],[4,5,6],[5,6,7],[1,2,3]]))

    def test_extrinsic_cv_split(self):
        trains, tests, validations = extrinsic_cv_split((self.source, self.target), sorted_vocab=True)
        self.assertEqual(trains[0][0], [c for c in 'bdefi'])
        self.assertEqual(trains[0][0], trains[1][0])
        self.assertEqual(tests[0][0], [])
        self.assertEqual(tests[1][0], tests[0][0])
        self.assertEqual(validations[0][0], [c for c in 'acjwx'])
        self.assertEqual([c for c in 'acjwx'], validations[1][0])
        self.assertEqual(trains[0][1].tolist(), [[2,3,4],[4,5,6],[5,6,7],[1,2,3],[2,3,4]])
        self.assertEqual(trains[1][1].tolist(), [[2,3,4],[3,4,5],[4,5,6],[5,6,7],[2,3,4]])
        self.assertEqual(tests[0][1].tolist(), [])
        self.assertEqual(tests[1][1].tolist(), [])
        self.assertEqual(validations[0][1].tolist(), [[1,2,3],[3,4,5],[3,4,5],[4,5,6],[5,6,7]])
        self.assertEqual(validations[1][1].tolist(), [[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0]])


if __name__ == '__main__':
    unittest.main()
