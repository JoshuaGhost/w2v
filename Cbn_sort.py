from numpy import array, ndarray, dot, diag, newaxis, sqrt, float64
from numpy.random import rand, random_integers
from gensim.models import Word2Vec
from pdb import set_trace as bp
from config import *


def get_inter_embd(models):
	ebds = []
	vocab_share = models[0].vocab.keys()
	for model in models:
		ebd = []
		for word in word_intersect:
			ebd.append(model.syn0[model.vocab[word].index])
		ebds.append(ebd)
	ebds = array(ebds)
	return ebds


def sort_align(embeddings, num_sample_words, abs_sort = False):
	num_word = embeddings.shape[1]
	dim = embeddings.shape[0]
	syn0_out = ndarray((5, num_word, dim))
	
	for ebd_idx, ebd in enumerate(embeddings):
		sample_word_idx = random_integers(0, num_word-1, (num_sample_words))
		rank_weight = array([0]*dim)
		
		for word_idx in sample_word_idx:
			vec = ebd[word_idx]
			abselement2orgidx = dict(zip(abs(vec) if abs_sort else vec, [i for i in range(dim)]))
			
			for idx, sorted_abselement in enumerate(sorted(abs(vec) if abs_sort else vec, reverse = True)):
				if vec[abselement2orgidx[sorted_abselement]] == sorted_abselement:
					rank_weight[abselement2orgidx[sorted_abselement]] += idx
				else:
					rank_weight[abselement2orgidx[sorted_abselement]] -= idx
		
		rank_weight2orig_idx = dict(zip(rank_weight, [i for i in range(dim)]))
		sorted_rank_weight = sorted(rank_weight)
		new_syn0 = ndarray((num_word, dim))
		for i in range(num_word):
			for j in range(dim):
				new_syn0[i][j] = ebd[i][rank_weight2orig_idx[sorted_rank_weight[j]]]
		syn0_out[ebd_idx, ..., ...] = new_syn0
	return syn0_out.sum(0)


class Cbn_sort(object):
	"""docstring for Sort_combiner"""
	def __init__(self, num_sample_words, abs_sort, test_mode):
		super(Sort_combiner, self).__init__()
		self.num_sample_words = num_sample_words
		self.abs_sort = abs_sort
		self.test_mode = test_mode


	def combine(models):

		embeddings = get_inter_embd(models)

		model_out = deepcopy(models[0])
		model_out.syn0 = sort_align(embeddings, self.num_sample_words, self.abs_sort)
		model_out.syn0norm = (model_out.syn0 / sqrt((model_out.syn0 ** 2).sum(-1))[..., newaxis]).astype(float64)
	
		temp_vocab = deepcopy(model_out.vocab)
		model_out.vocab = {}
		for word_idx, word in enumerate(word_intersect):
			model_out.vocab[word] = deepcopy(temp_vocab[word])
			model_out.vocab[word].index = word_idx
			model_out.index2word[word_idx] = word
		return model_out