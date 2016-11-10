from numpy import array, ndarray, dot, diag, newaxis, sqrt, float64
from numpy.random import rand, random_integers
from gensim.models import Word2Vec
from pdb import set_trace as bp


EBD_DIM = 100
LOSS_THRESHOLD = 1.
LEARNING_RATE = .015
NUM_SAMPLE_WORD = 500
MODEL_FOLDER = "/home/assassin/workspace/master_thesis/model/multiple_models/"


add = lambda x,y:x+y
mul = lambda x,y:x*y
sqr = lambda x: x**2


def cal_loss(ebd, W, ebd_standard):
	ebdy = dot(ebd, W)
	sqs = (ebdy-ebd_standard)**2
	return reduce(add, [reduce(add, sq) for sq in sqs])/num_word


def unarify(X):
	return (X / sqrt((X ** 2).sum(-1))[..., newaxis]).astype(float64)


models = [Word2Vec.load(MODEL_FOLDER+'multi_models_ebd_100_vvth_50_no_'+str(idx)+'.w2v') for idx in range(5)]
for model in models:
	model.init_sims(replace = True)



word_intersect = list(reduce(lambda x,y:x&y,[set(md.vocab.keys()) for md in models]))
num_word = len(word_intersect)


ebds = []
for model in models:
	ebd = []
	for word in word_intersect:
		ebd.append(model.syn0norm[model.vocab[word].index])
	ebds.append(ebd)
ebds = array(ebds)

syn0_out = ndarray((5, num_word, EBD_DIM))

for ebd_idx, ebd in enumerate(ebds):
	sample_word_idx = random_integers(0, num_word-1, (NUM_SAMPLE_WORD))
	rank_weight = array([0]*EBD_DIM)
	for word_idx in sample_word_idx:
		vec = ebd[word_idx]
		abselement2orgidx = dict(zip(abs(vec), [i for i in range(EBD_DIM)]))
		for idx, sorted_abselement in enumerate(sorted(abs(vec), reverse = True)):
			if vec[abselement2orgidx[sorted_abselement]] == sorted_abselement:
				rank_weight[abselement2orgidx[sorted_abselement]] += idx
			else:
				rank_weight[abselement2orgidx[sorted_abselement]] -= idx
		
	rw2origidx = dict(zip(rank_weight, [i for i in range(EBD_DIM)]))
	sortedrw = sorted(rank_weight)
	new_syn0 = ndarray((num_word, EBD_DIM))
	for i in range(num_word):
		for j in range(EBD_DIM):
			new_syn0[i][j] = ebd[i][rw2origidx[sortedrw[j]]]

	syn0_out[ebd_idx, ..., ...] = new_syn0


from copy import deepcopy
model_out = deepcopy(models[0])
model_out.syn0 = syn0_out.sum(0)
for word_idx, word in enumerate(word_intersect):
	model_out.vocab[word].index = word_idx
	model_out.index2word[word_idx] = word

print model_out.most_similar("this")
model_out.save(MODEL_FOLDER+"combined.w2v")
