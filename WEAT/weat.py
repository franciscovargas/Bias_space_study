import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from itertools import combinations,  filterfalse, permutations
from gensim.models.keyedvectors import KeyedVectors
import random
import matplotlib.pyplot as plt
from scipy.special  import comb


def swAB(W, A, B, similarity=cosine_similarity):
    """Calculates differential cosine-similarity between word vectors in W, A and W, B
     Arguments
              W, A, B : n x d matrix of word embeddings stored row wise
    """
    WA = similarity(W,A)
    WB = similarity(W,B)

    #Take mean along columns
    WAmean = np.mean(WA, axis = 1)
    WBmean = np.mean(WB, axis = 1)

    return (WAmean - WBmean)


def test_statistic(X, Y, A, B, similarity=cosine_similarity):
    """Calculates test-statistic between the pair of association words and target words
     Arguments
              X, Y, A, B : n x d matrix of word embeddings stored row wise
     Returns
              Test Statistic
    """
    return sum(swAB(X, A, B, similarity)) - sum(swAB(Y, A, B, similarity))


def weat_effect_size(X, Y, A, B, embd, similarity=cosine_similarity):
    """Computes the effect size for the given list of association and target word pairs
     Arguments
              X, Y : List of association words
              A, B : List of target words
              embd : Dictonary of word-to-embedding for all words
     Returns
              Effect Size
    """

    Xmat = np.array([embd[w.lower()] for w in X if w.lower() in embd])
    Ymat = np.array([embd[w.lower()] for w in Y if w.lower() in embd])
    Amat = np.array([embd[w.lower()] for w in A if w.lower() in embd])
    Bmat = np.array([embd[w.lower()] for w in B if w.lower() in embd])

    XuY = list(set(X).union(Y))
    XuYmat = []
    for w in XuY:
        if w.lower() in embd:
          XuYmat.append(embd[w.lower()])
    XuYmat = np.array(XuYmat)


    d = (np.mean(swAB(Xmat,Amat,Bmat, similarity)) - np.mean(swAB(Ymat,Amat,Bmat,similarity)))\
        /np.std(swAB(XuYmat, Amat, Bmat, similarity))
    # import pdb; pdb.set_trace()
    return d


def random_permutation(iterable, r=None):
    """Returns a random permutation for any iterable object"""
    pool = tuple(iterable)
    r = len(pool) if r is None else r
    return tuple(random.sample(pool, r))


def weat_p_value(X, Y, A, B, embd, sample = 1000, similarity=cosine_similarity):
    """Computes the one-sided P value for the given list of association and target word pairs
     Arguments
              X, Y : List of association words
              A, B : List of target words
              embd : Dictonary of word-to-embedding for all words
              sample : Number of random permutations used.
     Returns
    """
    size_of_permutation = min(len(X), len(Y))
    X_Y = X + Y
    test_stats_over_permutation = []

    Xmat = np.array([embd[w.lower()] for w in X if w.lower() in embd])
    Ymat = np.array([embd[w.lower()] for w in Y if w.lower() in embd])
    Amat = np.array([embd[w.lower()] for w in A if w.lower() in embd])
    Bmat = np.array([embd[w.lower()] for w in B if w.lower() in embd])

    if not sample:
        permutations = combinations(X_Y, size_of_permutation)
        # permutations = permutations(X_Y)
        print(comb(len(X_Y), size_of_permutation))
    else:
        permutations = [random_permutation(X_Y, size_of_permutation) for s in range(sample)]

    for Xi in permutations:
        Yi = filterfalse(lambda w:w in Xi, X_Y)
        Ximat = np.array([embd[w.lower()] for w in Xi if w.lower() in embd])
        Yimat = np.array([embd[w.lower()] for w in Yi if w.lower() in embd])
        test_stats_over_permutation.append(test_statistic(Ximat, Yimat, Amat, Bmat, similarity))


    unperturbed = test_statistic(Xmat, Ymat, Amat, Bmat, similarity)
    # print(max(test_stats_over_permutation), min(test_stats_over_permutation))
    # print(unperturbed)
    is_over = np.array([o > unperturbed for o in test_stats_over_permutation])
    # print(is_over.sum() )
    return is_over.sum() / is_over.size
