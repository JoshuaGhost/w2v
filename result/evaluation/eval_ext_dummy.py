from __future__ import division
from __future__ import print_function

import argparse
import os
import glob

from six import string_types, text_type
from gensim.models import Word2Vec
from config import (ebd_dim_begin, ebd_dim_end,	ebd_dim_step,
                    vocab_freq_thres_begin, vocab_freq_thres_end, vocab_freq_thres_step)


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

word_pairs = []
category = []
category_high_level = []
files = wikipedia_dict + wordnet
for file_name in files:
	c = os.path.basename(file_name).split(".")[0]
	c = c[c.index('-')+1:]
	with open(file_name, 'r') as f:
		for l in f.readlines():
			word_pairs.append(standardize_string(l).split())
			category.append(c)
			category_high_level.append('wikipedia-dict' if file_name in wikipedia_dict else 'wwordnet')


wordnet_categories = {'Antonym',
     'Attribute',
     'Causes',
     'DerivedFrom',
     'Entails',
     'HasContext',
     'InstanceOf',
     'IsA',
     'MadeOf',
     'MemberOf',
     'PartOf',
     'RelatedTo',
     'SimilarTo'}

wikipedia_categories = {'adjective-to-adverb',
     'all-capital-cities',
     'city-in-state',
     'comparative',
     'currency',
     'man-woman',
     'nationality-adjective',
     'past-tense',
     'plural-nouns',
     'plural-verbs',
     'present-participle',
     'superlative'}


model_files = glob.glob(os.path.join(model_dir, "word2vec*.w2v"))
with open('absolute_performance.csv', 'w+') as f:
	f.write('embedding dimension, vocabulary threshold, performance, recall\n')


def rank_similarity (model, x, y, topn = 20):
	top20 = model.most_similar(positive = [x], topn = topn)
	rank = 1
	if not y in dict(top20):
		rank = 21
	else:
		rank = dict(zip(map(lambda a: a[0], top20), [i for i in range(1,21)]))[y]
	return (1.0/rank)


for model_file in model_files:
	print('evaluating ', model_file)
	model = Word2Vec.load(model_file)
	num_word_pair = 0
	sim = 0.0
	for word_pair in word_pairs:
		if word_pair[0] in model.vocab and word_pair[1] in model.vocab:
			num_word_pair += 1
			if num_word_pair % 100 == 0:
				print (num_word_pair, ' pairs of word have been evaluated')
			sim += (rank_similarity(model, word_pair[0], word_pair[1]) + 
				    rank_similarity(model, word_pair[1], word_pair[0])) / 2.0

	filename = os.path.basename(model_file).split('.')[0]
	(_, ebd, vvth) = filename.split('_')
	ebd = ebd[3:]
	vvth = vvth[4:]
	with open("absolute_performance.csv", "a+") as f:
		f.write(', '.join((ebd,vvth,str(sim/num_word_pair), str(num_word_pair)))+'\n')
