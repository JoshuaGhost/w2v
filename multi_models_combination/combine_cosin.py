from numpy import array, ndarray, dot
from gensim.models import Word2Vec
from math import sqrt
EBD_DIM = 100

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
	for i in range(len(Y)):
		for j in range(i+1, len(Y)):
			vec_pairs = zip(Y[i],Y[j])
			cosin_sim_for_all_words = map(cosin_sim, vec_pairs)
			mean_cosin_sim = reduce(add, cosin_sim_for_all_words)/num_word
		goodness += mean_cosin_sim
	return goodness


models = [Word2Vec.load('multi_models_ebd_100_vvth_50_no_'+str(idx)+'.w2v') for idx in range(5)]
word_intersect = reduce(lambda x,y:x&y,[set(md.vocab.keys()) for md in models])
num_word = len(word_intersect)
ebd_for_all_models = [array([model.syn0[model.vocab[word].index for word in word_intersect] for model in models])]

Ws = ndarray((5, EBD_DIM, EBD_DIM))

goodness = cal_goodness(ebd_for_all_models, Ws)

array_pairs = [[model.syn0norm[model.vocab[word].index], model2.syn0norm[model2.vocab[word].index]] for word in word_intersect]
loss = lambda x: reduce(lambda lv1, lv2: lv1+lv2, map(lambda e_pair: (e_pair[0]-e_pair[1])**2))
reduce(lambda x,y: x+y, map(loss, array_pairs))/len(array_pairs)