# Code taken shamelessly from https://github.com/nmrksic/eval-multilingual-simlex/blob/master/evaluate.py
# Adapted to work with python 3
import numpy
import codecs
import sys
import time
import random
import math
import os
from copy import deepcopy
import json
from numpy.linalg import norm
from numpy import dot
from scipy.stats import spearmanr
from sklearn.externals import joblib
from os.path import join
from tqdm import tqdm

from L101_utils.data_paths import (bolu_professions, bolu_googlew2v, googlew2v, glove,
                                   bolu_gender_specific, bolu_equalize_pairs,
                                   bolu_definitional_pairs, model, data, googlew2vtxt)

lp_map = {}
lp_map["english"] = u"en_"
lp_map["german"] = u"de_"
lp_map["italian"] = u"it_"
lp_map["russian"] = u"ru_"


def normalise_word_vectors(word_vectors, norm=1.0):
    """
    This method normalises the collection of word vectors provided in the word_vectors dictionary.
    """
    print("normalising")
    for word in word_vectors:
        word_vectors[word] /= math.sqrt((word_vectors[word]**2).sum() + 1e-6)
        word_vectors[word] = word_vectors[word] * norm
    print("normalised")
    return word_vectors


def load_word_vectors(file_destination=googlew2vtxt, language="english"):
    """
    This method loads the word vectors from the supplied file destination.
    It loads the dictionary of word vectors and prints its size and the vector dimensionality.
    """
    print("Loading word vectors from", file_destination)
    word_dictionary = {}

    try:
        f = codecs.open(file_destination, 'r', 'utf-8')

        for ij, line in enumerate(tqdm(f)):
            if ij == 0: continue
            line = line.split(" ", 1)
            key = line[0].lower()
            if lp_map[language] not in key:
                key = lp_map[language] + key
            try:
                transformed_key = str(key)
            except:
                print( "CANT LOAD", transformed_key)
                raise

            try:
                word_dictionary[transformed_key] = numpy.fromstring(line[1], dtype="float32", sep=" ")
                assert word_dictionary[transformed_key].shape[0] == 300
            except:
                print(transformed_key)
    except:
        print( "Word vectors could not be loaded from:", file_destination)
        raise
        return {}

    print( len(word_dictionary), "vectors loaded from", file_destination)

    return normalise_word_vectors(word_dictionary)


def distance(v1, v2, normalised_vectors=True):
    """
    Returns the cosine distance between two vectors.
    If the vectors are normalised, there is no need for the denominator, which is always one.
    """
    if normalised_vectors:
        return 1 - dot(v1, v2)
    else:
        return 1 - dot(v1, v2) / ( norm(v1) * norm(v2) )


def simlex_analysis(word_vectors, language="german", source="simlex"):
    """
    This method computes the Spearman's rho correlation (with p-value) of the supplied word vectors.
    The method also prints the gold standard SimLex-999 ranking to results/simlex_ranking.txt,
    and the ranking produced using the counter-fitted vectors to results/counter_ranking.txt
    """
    pair_list = []
    fread_simlex=codecs.open(data + "/simlex-" + language + ".txt", 'r', 'utf-8')


    line_number = 0
    for line in fread_simlex:

        if line_number > 0:
            tokens = line.split()
            word_i = tokens[0].lower()
            word_j = tokens[1].lower()
            score = float(tokens[2])

            word_i = lp_map[language] + word_i
            word_j = lp_map[language] + word_j

            if word_i in word_vectors and word_j in word_vectors:
                pair_list.append( ((word_i, word_j), score) )
            else:
                pass
        line_number += 1

    pair_list.sort(key=lambda x: - x[1])

    coverage = len(pair_list)

    extracted_list = []
    extracted_scores = {}
    # print(pair_list)
    for (x,y) in pair_list:

        (word_i, word_j) = x
        try:
            current_distance = corrected_cosine_dist(word_vectors[word_i], word_vectors[word_j])
        except:
            import pdb; pdb.set_trace()
        extracted_scores[(word_i, word_j)] = current_distance
        extracted_list.append(((word_i, word_j), current_distance))

    extracted_list.sort(key=lambda x: x[1])

    spearman_original_list = []
    spearman_target_list = []

    for position_1, (word_pair, score_1) in enumerate(pair_list):
        score_2 = extracted_scores[word_pair]
        position_2 = extracted_list.index((word_pair, score_2))
        spearman_original_list.append(position_1)
        spearman_target_list.append(position_2)
    # print(spearman_original_list, spearman_target_list)
    spearman_rho = spearmanr(spearman_original_list, spearman_target_list)
    print(spearman_rho)
    return round(spearman_rho[0], 3), coverage


def main():
    """
    The user can provide the location of the config file as an argument.
    If no location is specified, the default config file (experiment_parameters.cfg) is used.
    """
    try:
        word_vector_location = googlew2vtxt
        language = "english"
        word_vectors = load_word_vectors(word_vector_location, language)
    except:
        print( "USAGE: python code/simlex_evaluation.py word_vector_location language")
        raise


    print("\n============= Evaluating word vectors for language:", language, " =============\n")

    simlex_score, simlex_coverage = simlex_analysis(word_vectors, language)
    print( "SimLex-999 score and coverage:", simlex_score, simlex_coverage)


if __name__=='__main__':
    sk_model = joblib.load(join(model, "joblib_kpca_lap_rkhsfix_model_k_1.pkl"))
    corrected_cosine_dist = lambda X ,Y: 1 - sk_model.corrected_cosine_similarity(X.reshape(1, -1),
                                                                                  Y.reshape(1, -1))[0,0]
    main()
