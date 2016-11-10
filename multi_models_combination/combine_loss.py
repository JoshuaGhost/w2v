from numpy import array, ndarray, dot, diag, newaxis, sqrt, float64
from numpy.random import rand, random_integers
from gensim.models import Word2Vec
from pdb import set_trace as bp


EBD_DIM = 100
LOSS_THRESHOLD = 1.
LEARNING_RATE = .015
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


model_standard = Word2Vec.load(MODEL_FOLDER+'multi_models_ebd_100_vvth_50_no_0.w2v')
model_standard.init_sims(replace=False)
models = [Word2Vec.load(MODEL_FOLDER+'multi_models_ebd_100_vvth_50_no_'+str(idx)+'.w2v') for idx in range(1, 5)]
for model in models:
	model.init_sims(replace=False)


word_intersect = reduce(lambda x,y:x&y,[set(md.vocab.keys()) for md in models+[model_standard]])
num_word = len(word_intersect)


ebds = []
for model in models:
	ebd = []
	for word in word_intersect:
		ebd.append(model.syn0norm[model.vocab[word].index])
	ebds.append(ebd)
ebds = array(ebds)

ebd_standard = []
for word in word_intersect:
	ebd_standard.append(model_standard.syn0norm[model_standard.vocab[word].index])
ebd_standard = array(ebd_standard)


Ws = []
for ebd in ebds:
	W = array([[1./EBD_DIM for j in range(EBD_DIM)] for i in range(EBD_DIM)])
	#W = rand(EBD_DIM, EBD_DIM)*1e-100
	print W
	loss = cal_loss(ebd, W, ebd_standard)
	print loss
	while loss > LOSS_THRESHOLD:
		for idx, vec in enumerate(ebd):
			d1 = 2*(dot(vec, W)-ebd_standard[idx])
			d2 = array([W[...,i]*d1[i] for i in range(len(d1))])
			dirivation = d2.T
			W = W*(1-LEARNING_RATE)-dirivation*LEARNING_RATE
		W = unarify(W)
		loss = cal_loss(ebd, W, ebd_standard)
		print loss
		print W
	print("=====================================")
	print W
	print("=====================================")
	Ws.append(W)
	continue
	
