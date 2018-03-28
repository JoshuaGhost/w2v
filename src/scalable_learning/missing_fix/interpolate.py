import numpy as np
from scalable_learning.extrinsic_evaluation.web.embedding import Embedding
from scalable_learning.extrinsic_evaluation.web.vocabulary import OrderedVocabulary
from scipy.linalg import svd, inv, eigh


def cca(webs_train, webs_test, webs_predict):
    web0, web1 = webs_train
    web0.append(webs_test[0])
    web1.append(webs_test[1])
    vecs0 = np.asarray(web0.vectors)
    vecs1 = np.asarray(web1.vectors)

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
    vecs_predict0 = np.asarray(webs_predict[0].vectors).dot(A)
    vecs_predict1 = np.asarray(webs_predict[1].vectors).dot(B)

    web0.vectors = vecs0
    web1.vectors = vecs1
    web_predict0 = Embedding(vocabulary=webs_predict[0].vocabulary, vectors=vecs_predict0)
    web_predict1 = Embedding(vocabulary=webs_predict[1].vocabulary, vectors=vecs_predict1)

    web0.append(web_predict0)
    web0.append(web_predict1)
    web1.append(web_predict0)
    web1.append(web_predict1)
    return web0, web1


def orthogonal_procrustes(webs_train, webs_test, webs_predict):
    web0, web1 = webs_train
    web0.append(webs_test[0])
    web1.append(webs_test[1])
    vecs0 = np.asarray(web0.vectors)
    vecs1 = np.asarray(web1.vectors)

    M120 = vecs1.T.dot(vecs0)
    U, _, Vh = svd(M120)
    R120 = U.dot(Vh)
    predict_vec0 = np.asarray(webs_predict[1].vectors).dot(R120)
    web0_predict = Embedding(vocabulary=webs_predict[1].vocabulary, vectors=predict_vec0)

    M021 = vecs0.T.dot(vecs1)
    U, _, Vh = svd(M021)
    R021 = U.dot(Vh)
    predict_vec1 = np.asarray(webs_predict[0].vectors).dot(R021)
    web1_predict = Embedding(vocabulary=webs_predict[0].vocabulary, vectors=predict_vec1)

    web0.append(webs_predict[0])
    web0.append(web0_predict)
    web1.append(webs_predict[1])
    web1.append(web1_predict)
    return web0, web1


def affine_transform(webs_train, webs_test, webs_predict):
    web0, web1 = webs_train
    web0.append(webs_test[0])
    web1.append(webs_test[1])
    vecs0 = np.asarray(web0.vectors)
    vecs1 = np.asarray(web1.vectors)

    U, S, Vh = svd(vecs1, full_matrices=True)
    S_inv = np.hstack((np.diag(1/S), np.zeros((Vh.shape[0], U.shape[0]-Vh.shape[0]))))
    T = Vh.T.dot(S_inv).dot(U.T).dot(vecs0)
    # T = inv(vecs1.T.dot(vecs1)).dot(vecs1.T).dot(vecs0)
    predict_vec0 = np.asarray(webs_predict[1].vectors).dot(T)
    web0_predict = Embedding(vocabulary=webs_predict[1].vocabulary, vectors=predict_vec0)

    U, S, Vh = svd(vecs0, full_matrices=True)
    S_inv = np.hstack((np.diag(1/S), np.zeros((Vh.shape[0], U.shape[0]-Vh.shape[0]))))
    T = Vh.T.dot(S_inv).dot(U.T).dot(vecs1)
    # T = inv(vecs0.T.dot(vecs0)).dot(vecs0.T).dot(vecs1)
    predict_vec1 = np.asarray(webs_predict[0].vectors).dot(T)
    web1_predict = Embedding(vocabulary=webs_predict[0].vocabulary, vectors=predict_vec1)

    web0.append(webs_predict[0])
    web0.append(web0_predict)
    web1.append(webs_predict[1])
    web1.append(web1_predict)
    return web0, web1


def fill_zero(webs_train, webs_test, webs_predict):
    web0, web1 = webs_train
    web_predict1 = Embedding(vocabulary=webs_predict[0].vocabulary, vectors=np.zeros_like(webs_predict[0].vectors))
    web_predict0 = Embedding(vocabulary=webs_predict[1].vocabulary, vectors=np.zeros_like(webs_predict[1].vectors))

    web0.append(webs_test[0])
    web0.append(webs_predict[0])
    web0.append(web_predict0)

    web1.append(webs_test[1])
    web1.append(webs_predict[1])
    web1.append(web_predict1)

    return web0, web1
