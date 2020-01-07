# Taken from https://github.com/gonenhila/gender_bias_lipstick/blob/master/source/remaining_bias_2016.ipynb
import os
import json
from L101_utils.data_paths import (bolu_professions, bolu_googlew2v, googlew2v, glove,
                                   bolu_gender_specific, bolu_equalize_pairs,
                                   bolu_definitional_pairs, model)
from sklearn.metrics.pairwise import cosine_similarity
from L101_utils.mock_model import MockModel
import scipy
from collections import Counter
import numpy as np
import string
from tqdm import tqdm
import codecs
import numpy as np
from numpy import linalg as LA
import json
from sklearn.externals import joblib
from os.path import join


from sklearn import svm
from random import shuffle
import random
random.seed(10)

def make_gender_bias_dict(emb=None, max_words=50000):

    if emb is None:
        print("loading old embs")
        emb = MockModel.from_file(glove, mock=False)
        print("loaded")
    he = emb["he"]
    she = emb["she"]
    vecs, vocab = emb.vectors[:max_words, :], emb.index2word[:max_words]
    print("normalising vecs")
    vecs = vecs / np.linalg.norm(vecs,  axis=1)[..., None]
    print("normalised vecs")

    male =  cosine_similarity(vecs, he.reshape(1,-1))
    female =  cosine_similarity(vecs, she.reshape(1,-1))
    del emb
    print(male.shape, female.shape)
    return male.flatten() - female.flatten(), vecs, vocab


def main(vec_path=googlew2v):


    sk_model = joblib.load(join(model, "joblib_kpca_lap_rkhsfix_model_glove_k_1.pkl"))
    print((sk_model.lambdas_.flatten() != 0).sum())
    corrected_cosine = lambda X ,Y: sk_model.corrected_cosine_similarity(X, Y)

    def rbf_on_RKHS(X,Y, similarity=corrected_cosine):
        Kxx = np.diag(similarity(X,X)).reshape(X.shape[0], 1)
        Kyy = np.diag(similarity(Y,Y)).reshape(1, Y.shape[0])
        Kxy = similarity(X,Y)
        gamma =  1.0 / X.shape[1]

        return np.exp( -gamma * (Kxx - 2 * ( Kxy) + Kyy ) )

    gender_bias_bef, vecs, vocab = make_gender_bias_dict()
    size_train = 500
    size_test = 2000
    size = size_train + size_test

    indxs = np.argsort(gender_bias_bef)

    females = [vecs[i].copy() for i in indxs[:size]]
    males = [vecs[i].copy() for i in indxs[-size:]]
    del vecs
    print(np.array(vocab)[indxs[:size]])
    print((np.array(vocab)[indxs[size:]])[::-1])
    shuffle(females)
    shuffle(males)

    X_train = males[:size_train] + females[:size_train]
    Y_train = [1] * size_train + [0] * size_train
    X_test =  males[size_train:] + females[size_train:]
    Y_test = [1] * size_test + [0] * size_test

    clf = svm.SVC(kernel=rbf_on_RKHS, gamma="auto")
    # clf = svm.SVC(kernel="rbf", gamma="auto")
    clf.fit(X_train, Y_train)

    preds = clf.predict(X_test)

    accuracy = [1 if y==z else 0 for y,z in zip(preds, Y_test)]
    print( 'accuracy:', float(sum(accuracy)) / len(accuracy))


if __name__ == '__main__':
    main(vec_path=glove)
