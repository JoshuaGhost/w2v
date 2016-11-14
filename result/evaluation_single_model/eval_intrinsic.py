from __future__ import division
from __future__ import print_function

import argparse
import os
import glob

from six import string_types, text_type
from gensim.models import Word2Vec


_delchars = [chr(c) for c in range(256)]
_delchars = [x for x in _delchars if not x.isalnum()]
_delchars.remove('\t')
_delchars.remove(' ')
_delchars.remove('-')
_delchars.remove('_')
_delchars = ''.join(_delchars)
_delchars_table = dict((ord(char), None) for char in _delchars)


def standardize_string(s, clean_words=True, lower=True, language="english"):
	assert isinstance(s, string_types)

	if not isinstance(s, text_type):
		s = text_type(s, "utf-8")

	if language == "english":
		s = (s.lower() if lower else s)
		s = (s.translate(_delchars_table) if clean_words else s)
		return s
	else:
		raise NotImplementedError("Not implemented standarization for other languages")


parser = argparse.ArgumentParser(description = 'Process some integers.')
parser.add_argument('-m', type = str)
parser.add_argument('-d', type = str)
args = vars(parser.parse_args())
model_dir = args['m']
data_dir = args['d']


assert os.path.exists(model_dir)
assert os.path.exists(data_dir)


wikipedia_dict = glob.glob(os.path.join(data_dir, "Pairs_from_Wikipedia_and_Dictionary/*.txt"))
wordnet = glob.glob(os.path.join(data_dir, "Pairs_from_WordNet/*.txt"))

words_query = set()
files = wikipedia_dict + wordnet
for file_name in files:
	c = os.path.basename(file_name).split(".")[0]
	c = c[c.index('-')+1:]
	with open(file_name, 'r') as f:
		for l in f.readlines():
			words_query.add(standardize_string(l).split()[0])


model_files = glob.glob(os.path.join(model_dir, "word2vec*.w2v"))
with open('intrinsic.csv', 'w+') as f:
	f.write('embedding dimension, vocabulary threshold, average intersect rate\n')


sim_set_min_vocabs = {}
model_min_vocab_name = 'word2vec_ebd100_vvth50.w2v'

for i in range(1):
    model_min_vocab = Word2Vec.load(model_dir+('/' if model_dir[-1] != '/' else '')+model_min_vocab_name)
    min_vocab = set(model_min_vocab.vocab) & words_query
    for word in min_vocab:
        sim_set_min_vocabs[word] = set([d[0] for d in model_min_vocab.most_similar(positive = [word], topn = 10)])


print ('queries have been loaded')


for model_file in model_files:
    model_basename = os.path.basename(model_file)
    if model_basename == model_min_vocab_name:
        continue
    vvth = model_basename[model_basename.index('vvth')+4:model_basename.index('.')]
    ebd  = model_basename[model_basename.index('ebd')+3:model_basename.index('_vvth')]

    print('evaluating ', model_file)
    model = Word2Vec.load(model_file)
    vocab = set(model.vocab) & min_vocab
    intersect_query = len(vocab)
    intersect_rate = map(lambda word: len(set(d[0] for d in model.most_similar(positive = [word], topn = 10))&sim_set_min_vocabs[word]) / 10.0, vocab)
    intersect_rate_total = reduce(lambda x, y: x+y, intersect_rate)

    with open("intrinsic.csv", "a+") as f:
        f.write(', '.join((ebd,vvth,str(intersect_rate_total/intersect_query)))+'\n')
