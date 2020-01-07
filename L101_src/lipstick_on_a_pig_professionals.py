import os
import json
from L101_utils.data_paths import bolu_professions, bolu_googlew2v, googlew2v, model, glove
from sklearn.metrics.pairwise import cosine_similarity
from L101_utils.mock_model import MockModel
import scipy
from collections import Counter
import numpy as np
from sklearn.externals import joblib
from os.path import join


def load_professions():
    professions_file = bolu_professions
    with open(professions_file, 'r') as f:
        professions = json.load(f)
    print('Loaded professions\n' +
          'Format:\n' +
          'word,\n' +
          'definitional female -1.0 -> definitional male 1.0\n' +
          'stereotypical female -1.0 -> stereotypical male 1.0')
    return professions


def compute_neighbors(vecs, thresh, max_words, similarity, threshold=False):
        thresh = float(thresh) # dang python 2.7!

        print("Computing neighbors")
        dots = similarity(vecs, vecs)
        print("computed sim")
        # if threshold:
        #     dots = scipy.sparse.csr_matrix( dots * (dots >= 1-thresh/2) )
        # rows, cols = dots.nonzero()
        # nums = list(Counter(rows).values())
        # print("Mean:", np.mean(nums) - 1)
        # print("Median:", np.median(nums) - 1)
        # rows, cols, vecs = zip(*[(i, j, vecs[i]-vecs[j]) for i, j, x in zip(rows, cols, dots.data) if i<j])
        # return rows, cols, np.array([v/np.linalg.norm(v) for v in vecs])
        return  dots


def make_gender_bias_dict(emb=None, max_words=20000):

    if emb is None:
        print("loading old vecs")
        emb = MockModel.from_file(googlew2v, mock=False)
        print("loaded")
    he = emb["he"]
    she = emb["she"]
    vecs, vocab = emb.vectors[:max_words, :], emb.index2word[:max_words]

    male =  cosine_similarity(vecs, he.reshape(1,-1))
    female =  cosine_similarity(vecs, she.reshape(1,-1))
    del emb
    print(male.shape, female.shape)
    return male.flatten() - female.flatten()


def best_analogies_dist_thresh(vec_path, similarity,
                               thresh=1, topn=500, max_words=20000):
        """Metric is cos(a-c, b-d) if |b-d|^2 < thresh, otherwise 0
        """
        emb = MockModel.from_file(vec_path, mock=False)
        print("loaded embs")
        v = emb["he"] - emb["she"]
        vecs, vocab = emb.vectors[:max_words, :], emb.index2word[:max_words]

        vecs = vecs / np.linalg.norm(vecs,  axis=1)[..., None]
        print("deleting emb")
        emb = None
        del emb
        print("deleted emb")
        dots = compute_neighbors(vecs, thresh, max_words, similarity)
        # scores = vecs.dot(v / np.linalg.norm(v))
        rowz = np.arange(dots.shape[0])
        print("started sorting")
        del vecs, vocab
        print("deleted")
        idx = np.argpartition(-dots, kth=105, axis=1)
        print("done sorting")
        idxs = idx[:,:105]
        del  idx

        gender_bias_dict = make_gender_bias_dict(max_words=max_words)

        male_counts = []
        female_counts = []
        for iword in range(idxs.shape[0]):
            top = idxs[iword, :100]
            m = 0
            f = 0
            for t in top:
                if gender_bias_dict[t] > 0:
                    m += 1
                else:
                    f += 1
            male_counts.append(m)
            female_counts.append(f)

        return scipy.stats.pearsonr(gender_bias_dict, male_counts)


def main():

    sk_model = joblib.load(join(model, "joblib_kpca_lap_rkhsfix_model_k_1.pkl"))
    print((sk_model.lambdas_.flatten() != 0).sum())
    corrected_cosine = lambda X ,Y: sk_model.corrected_cosine_similarity(X, Y)
    print(" Models")
    # vec_path = bolu_googlew2v
    vec_path = googlew2v
    professions = load_professions()
    profession_scores = [ (p[1], p[2]) for p in professions]
    profession_words = [p[0] for p in professions]
    print("loaded profs")
    ans = best_analogies_dist_thresh(vec_path, similarity=corrected_cosine)
    print(ans)


if __name__ == '__main__':
    main()
