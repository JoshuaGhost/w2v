from numpy import array, ndarray, dot, diag, newaxis, sqrt, float64
from numpy.random import rand, random_integers
from gensim.models import Word2Vec
from pdb import set_trace as bp
from config import *


def get_models(basename, num_models):
	return [Word2Vec.load(basename + str(idx) + '.w2v') for idx in range(1, num_models+1)]


def get_embeddings(models, word_intersect):
	ebds = []
	for model in models:
		ebd = []
		for word in word_intersect:
			ebd.append(model.syn0[model.vocab[word].index])
		ebds.append(ebd)
	ebds = array(ebds)
	return ebds


def sort_combine(embeddings, abs_sort = False):

	num_word = embeddings.shape[1]
	syn0_out = ndarray((5, num_word, EBD_DIM))
	

	for ebd_idx, ebd in enumerate(embeddings):
		sample_word_idx = random_integers(0, num_word-1, (NUM_SAMPLE_WORD))
		rank_weight = array([0]*EBD_DIM)
		
		for word_idx in sample_word_idx:
			vec = ebd[word_idx]
			abselement2orgidx = dict(zip(abs(vec) if abs_sort else vec, [i for i in range(EBD_DIM)]))
			
			for idx, sorted_abselement in enumerate(sorted(abs(vec) if abs_sort else vec, reverse = True)):
				if vec[abselement2orgidx[sorted_abselement]] == sorted_abselement:
					rank_weight[abselement2orgidx[sorted_abselement]] += idx
				else:
					rank_weight[abselement2orgidx[sorted_abselement]] -= idx
		
		rank_weight2orig_idx = dict(zip(rank_weight, [i for i in range(EBD_DIM)]))
		sorted_rank_weight = sorted(rank_weight)
		new_syn0 = ndarray((num_word, EBD_DIM))
		for i in range(num_word):
			for j in range(EBD_DIM):
				new_syn0[i][j] = ebd[i][rank_weight2orig_idx[sorted_rank_weight[j]]]
		syn0_out[ebd_idx, ..., ...] = new_syn0
	return syn0_out.sum(0)


def combine_models(abs_sort):
	from copy import deepcopy
	models = get_models(MODEL_FOLDER+MODLE_BASENAME, 5)
	word_intersect = list(reduce(lambda x,y:x&y,[set(md.vocab.keys()) for md in models]))
	num_word = len(word_intersect)

	embeddings = get_embeddings(models, word_intersect)

	model_out = deepcopy(models[0])
	model_out.syn0 = sort_combine(embeddings, abs_sort)
	model_out.syn0norm = (model_out.syn0 / sqrt((model_out.syn0 ** 2).sum(-1))[..., newaxis]).astype(float64)
	
	temp_vocab = deepcopy(model_out.vocab)
	model_out.vocab = {}
	for word_idx, word in enumerate(word_intersect):
		model_out.vocab[word] = deepcopy(temp_vocab[word])
		model_out.vocab[word].index = word_idx
		model_out.index2word[word_idx] = word
	return model_out
