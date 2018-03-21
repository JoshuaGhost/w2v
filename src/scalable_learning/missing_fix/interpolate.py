import numpy as np
from scalable_learning.extrinsic_evaluation.web.embedding import Embedding
from scalable_learning.extrinsic_evaluation.web.vocabulary import OrderedVocabulary
from scipy.linalg import svd, inv, eigh

def global_transformation(web1, web2, pct_train):
    pass


def cca(webs_train, webs_test=None, webs_validation=None):
    web0, web1 = webs_train
    web0.append(webs_test[0])
    web1.append(webs_test[1])
    vecs0 = np.asarray(web0.vectors)
    vecs1 = np.asarray(web1.vectors)
    vecs_val0 = np.asarray(webs_validation[0].vectors)
    vecs_val1 = np.asarray(webs_validation[1].vectors)

    sigma_00 = vecs0.T.dot(vecs0)
    sigma_11 = vecs1.T.dot(vecs1)
    sigma_01 = vecs0.T.dot(vecs1)
    sigma_00_i = inv(sigma_00)
    sigma_11_i = inv(sigma_11)
    sigma_A = sigma_00_i.dot(sigma_01.dot(sigma_11_i.dot(sigma_01.T)))
    sigma_B = sigma_11_i.dot(sigma_01.T.dot(sigma_00_i.dot(sigma_01)))

    _, A = eigh(sigma_A)  # A, B are orthogonal matrix thus multiplication of them is a basis-change
    _, B = eigh(sigma_B)  # A, B are orthogonal matrix thus multiplication of them is a basis-change

    vecs0 = vecs0.dot(A)
    vecs1 = vecs1.dot(B)
    vecs_val0 = vecs_val0.dot(A)
    vecs_val1 = vecs_val1.dot(B)
    webs_train[0].vectors = vecs0
    webs_train[1].vectors = vecs1
    webs_validation[0].vectors = vecs_val0
    webs_validation[1].vectors = vecs_val1
    webs_train[0].append(webs_validation[0])
    webs_train[0].append(webs_validation[1])  # prediction using cca
    webs_train[1].append(webs_validation[1])
    webs_train[1].append(webs_validation[0])  # prediction using cca
    return webs_train


def orthogonal_procrustes(webs_train, webs_test=None, webs_validation=None):
    web0, web1 = webs_train

    web0.append(webs_test[0])
    web1.append(webs_test[1])
    vecs0 = np.array(web0.vectors)
    vecs1 = np.array(web1.vectors)

    M021 = vecs0.T.dot(vecs1)
    U, _, Vh = svd(M021)
    R021 = U.dot(Vh)
    predict_vec1 = np.array(webs_validation[0].vectors).dot(R021)
    web1_predict = Embedding(vocabulary=webs_validation[0].vocab, vectors=predict_vec1)

    M120 = vecs1.T.dot(vecs0)
    U, _, Vh = svd(M120)
    R120 = U.dot(Vh)
    predict_vec0 = np.array(webs_validation[1].vectors).dot(R120)
    web0_predict = Embedding(vocabulary=webs_validation[1].vocab, vectors=predict_vec0)

    web0.append(web0_predict)
    web1.append(web1_predict)
    return web0, web1


def fill_zero(webs_train, webs_test=None, webs_validation=None):
    web0, web1 = webs_train
    web_predict1 = Embedding(vocabulary=webs_validation[0].vocab, vectors=np.zeros_like(webs_validation[0].vectors))
    web_predict0 = Embedding(vocabulary=webs_validation[1].vocab, vectors=np.zeros_like(webs_validation[1].vectors))
    web0.append(webs_test[0])
    web0.append(web_predict0)
    web1.append(webs_test[1])
    web1.append(web_predict1)
    return web0, web1


def tcca(ms):
    pass


def gcca(ms):
    pass