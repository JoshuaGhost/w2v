from __future__ import division
from __future__ import print_function

import argparse
import os
import glob

from six import string_types, text_type
from gensim.models import Word2Vec

from utils import standardize_string, err


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


class Eval_wordnet_wiki(object):
	def __init__(self, ce_dir, exp_type, 
		benchmark_dir, ben_form, num_total_docs,
		num_sub_model, test):
		super(Eval_wordnet_wiki, self).__init__()
		self.ce_dir = ce_dir
		self.exp_type = exp_type
		self.ben_form = ben_form
		self.data_dir = benchmark_dir
		self.num_total_docs = -1 if num_total_docs is None else num_total_docs
		self.num_sub_model = -1 if exp_type==0 else (1 if num_sub_model is None else num_sub_model)
		self.test = test
		wikipedia_dict = glob.glob(os.path.join(self.data_dir, "Pairs_from_Wikipedia_and_Dictionary/*.txt"))
		wordnet = glob.glob(os.path.join(self.data_dir, "Pairs_from_WordNet/*.txt"))

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
		self.word_pairs = word_pairs
		self.category = category
		self.category_high_level = category_high_level


	def eval_ext(self, model, ebd_dim, min_count):
		with open(self.ce_dir+'extrinsic.csv', 'a+') as f:
			f.write('\n'+'='*22+'\n\n')
			f.write('type=%d,form=%d,num total docs=%d,num sub model=%d,dimension=%d,min_count=%d,test=%s\n'%
					(self.exp_type, self.ben_form, self.num_total_docs, self.num_sub_model, ebd_dim, min_count, self.test))
			f.write('data source, category, average ERR, occurence\n')
			occurence = {} 
			sim = {}
			gen = ((index, word_pair) for index, word_pair in enumerate(self.word_pairs) if set(word_pair).issubset(set(model.vocab)))

			for index, word_pair in gen:
				occurence[self.category[index]] = (occurence[self.category[index]] + 1) if occurence.has_key(self.category[index]) else 1
				avg_err_per_pair = (err(model, word_pair[0], word_pair[1]) + 
									err(model, word_pair[1], word_pair[0])) / 2.0
				sim[self.category[index]] = (sim[self.category[index]]+avg_err_per_pair) if sim.has_key(self.category[index]) else avg_err_per_pair
				if (index+1) % 100 == 0:
					print ('valuation of word pair num ', index+1, ' finished')
					if self.test:
						break

			for c in (wordnet_categories | wikipedia_categories) & set(sim.keys()) & set(occurence.keys()):
				to_print = (('wikipedia-dict' if c in wikipedia_categories else 'wordnet'), c, str(sim[c]/occurence[c]), str(occurence[c]))
				f.write(', '.join(to_print)+'\n')
			f.write('\n')
