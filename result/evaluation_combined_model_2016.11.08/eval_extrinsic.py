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

parser.add_argument('-d', type = str)
args = vars(parser.parse_args())

data_dir = args['d']


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
			category_high_level.append('wikipedia-dict' if file_name in wikipedia_dict else 'wordnet')


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


with open('extrinsic.csv', 'w+') as f:
	f.write('data source, category, average ERR, occurence\n')


def rank_similarity (model, x, y, topn = 20):
	top20 = model.most_similar(positive = [x], topn = topn)
	rank = 1
	if not y in dict(top20):
		rank = 2
	else:
		rank = dict(zip(map(lambda a: a[0], top20), [i for i in range(1,21)]))[y]
	return (1.0/rank)


from combine_sort import combine_models
models = [combine_models(abs_sort = True), combine_models(abs_sort = False)]
for model in models:
	print('evaluating ', model)
	occurence = {} 
	sim = {}
	gen = ((index, word_pair) for index, word_pair in enumerate(word_pairs) if set(word_pair).issubset(set(model.vocab)))
	for index, word_pair in gen:
		occurence[category[index]] = (occurence[category[index]] + 1) if occurence.has_key(category[index]) else 1
		avg_err_per_pair = (rank_similarity(model, word_pair[0], word_pair[1]) + 
							rank_similarity(model, word_pair[1], word_pair[0])) / 2.0
		sim[category[index]] = (sim[category[index]]+avg_err_per_pair) if sim.has_key(category[index]) else avg_err_per_pair
		if (index+1) % 100 == 0:
			print ('valuation of word num ', index+1, ' finished')

	with open("extrinsic.csv", "a+") as f:
		for c in (wordnet_categories | wikipedia_categories) & set(sim.keys()) & set(occurence.keys()):
			print (c)

			to_print = (('wikipedia-dict' if c in wikipedia_categories else 'wordnet'), c, str(sim[c]/occurence[c]), str(occurence[c]))
			f.write(', '.join(to_print)+'\n')

		f.write('\n')
