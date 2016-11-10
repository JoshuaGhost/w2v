from numpy import array, ndarray, dot
from numpy.random import rand, random_integers
from gensim.models import Word2Vec
from math import sqrt
EBD_DIM = 2
LOSS_THRESHOLD = .05
LEARNING_RATE = .05
MODEL_FOLDER = "/home/assassin/workspace/master_thesis/model/multiple_models_test/"

add = lambda x,y:x+y
mul = lambda x,y:x*y
sqr = lambda x: x**2


def cosin_sim(vec_pair, y = None):
	x = []
	if y is None:
		x = vec_pair[0]
		y = vec_pair[1]
	else:
		x = vec_pair

	cross = reduce(add, [e[0]*e[1] for e in zip(x,y)])
	x_l2 = sqrt(reduce(add, map(sqr, x)))
	y_l2 = sqrt(reduce(add, map(sqr, y)))
	return cross/(x_l2+y_l2)


def cal_goodness(Xs, Ws):
	Y = []
	for idx, x in enumerate(Xs):
		Y.append(dot(x, Ws[idx]))
	goodness = 0.
	centroid_per_word_per_model = [[]]
	for i in range(len(Y)):
		for j in range(i+1, len(Y)):
			vec_pairs = zip(Y[i],Y[j])
			cosin_sim_for_all_words = map(cosin_sim, vec_pairs)
			mean_cosin_sim = reduce(add, cosin_sim_for_all_words)/num_word
		goodness += mean_cosin_sim
	return goodness


def cal_loss(ebd, W, ebd_standard):
	ebdy = dot(ebd, W)
	sqs = (ebdy-ebd_standard)**2
	return reduce(add, [reduce(add, sq) for sq in sqs])/num_word


model_standard = Word2Vec.load(MODEL_FOLDER+'test0.w2v')
models = [Word2Vec.load(MODEL_FOLDER+'test'+str(idx)+'.w2v') for idx in range(1, 2)]
word_intersect = reduce(lambda x,y:x&y,[set(md.vocab.keys()) for md in models+[model_standard]])
num_word = len(word_intersect)
ebds = []
for model in models:
	ebd = []
	for word in word_intersect:
		ebd.append(model.syn0[model.vocab[word].index])
	ebds.append(ebd)
ebds = array(ebds)

ebd_standard = []
for word in word_intersect:
	ebd_standard.append(model_standard.syn0[model_standard.vocab[word].index])
ebd_standard = array(ebd_standard)

Ws = []

for ebd in ebds:
	W = ndarray((EBD_DIM, EBD_DIM))
	loss = cal_loss(ebd, W, ebd_standard)
	while loss > LOSS_THRESHOLD:
		for idx, vec in enumerate(ebd):
			d1 = 2*(dot(vec, W)-ebd_standard[idx])
			d2 = array([W[...,i]*d1[i] for i in range(len(d1))])
			dirivation = d2.T
			W = W-dirivation*LEARNING_RATE
		loss = cal_loss(ebd, W, ebd_standard)
		print loss
	print("=====================================")
	print W
	print("=====================================")
	Ws.append(W)
	continue
	
