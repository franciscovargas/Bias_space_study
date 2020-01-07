import json
import numpy as np
from L101_utils.data_paths import (wikift, bolu_gender_specific, smallc_glove,
                                   bolu_equalize_pairs, googlew2v, glove,
                                   bolu_definitional_pairs, model, data)
import numpy.linalg as la
from L101_utils.mock_model import MockModel
from sklearn.decomposition import PCA
from sklearn.externals import joblib
from os.path import join


def get_pc_projection_boluk_numpy(X, k=1, mean_rev=True):
    mean_rev = int(mean_rev)
    n, d = X.shape
    X = X - mean_rev * X.mean(axis=0)
    C = (X.T.dot(X) / n)
    D, V = la.eigh(C)
    V = V[:, :k]
    return V.dot(V.T), V


def get_pc_projection_boluk(X, k=1, save_model=True):

    n, d = X.shape
    pca = PCA(n_components=k)
    pca.fit(X)
    if save_model:
        joblib_file = join(model, f"joblib_pca_lin_model_k_{n_components}.pkl")
        joblib.dump(pca, joblib_file)
        print(f"Saved model at {joblib_file}")
        # exit()
    V = pca.components_[0].reshape(d, 1)
    return V.dot(V.T), V


def equalize_boluk(E, P, N=None, debug=True):
    I = np.eye(len(P))
    nu = (I - P).dot( E.mean(axis=0) )
    E = (E - E.mean(axis=0)[None, ...]).dot(P)

    E /= np.linalg.norm(E, axis=1)[..., None]

    v = np.linalg.norm(nu)
    fac = np.sqrt(1.0 - v**2)
    remb = nu +  fac * E

    return remb, E


def neutralise_boluk(X, P):
    I = np.eye(P.shape[0])
    out =  X.dot( (I - P).T )
    return out


def generate_subspace_projection(emb,
                                 pair_file=bolu_definitional_pairs,
                                 n_components=1):
    with open(pair_file, "r") as f:
        pairs = json.load(f)

    matrix = []
    for a, b in pairs:
        center = (emb.vectors[emb.vocab[a.lower()].index] + emb.vectors[emb.vocab[b.lower()].index])/2
        matrix.append(emb.vectors[emb.vocab[a.lower()].index] - center)
        matrix.append(emb.vectors[emb.vocab[b.lower()].index] - center)

    matrix = np.asarray(matrix)
    P, V = get_pc_projection_boluk(matrix, k=n_components)

    return P, V


def hard_debiase(emb,
                 gender_specific_file=bolu_gender_specific,
                 equalize_pair_file=bolu_equalize_pairs,
                 def_pair_file=bolu_definitional_pairs,
                 n_components=1,
                 norm=True,
                 mask=None):

    if norm:
        emb.vectors /= np.linalg.norm(emb.vectors,  axis=1)[..., None]

    P, V = generate_subspace_projection(emb, def_pair_file, n_components)
    assert (P == P.T).all()

    with open(gender_specific_file, "r") as f:
        gendered_words = set(json.load(f))
    #
    all_words = set(emb.vocab.keys())
    if mask is None: mask = all_words
    neutral_words = all_words - gendered_words
    neutral_words = list(set(mask) & neutral_words)

    word2index = [emb.vocab[k].index for k in neutral_words]

    neutral = emb.vectors[word2index,:]
    emb.vectors[word2index,:] = neutralise_boluk(neutral, P)

    with open(bolu_equalize_pairs, "r") as f:
        equalize_words = json.load(f)

    candidates = {x for e1, e2 in equalize_words for x in [(e1.lower(), e2.lower()),
                                                           (e1.title(), e2.title()),
                                                           (e1.upper(), e2.upper())]}
    print(candidates, "started equalising")
    for (e1, e2) in candidates:
        if (e1 in mask and e2 in mask):
            word2index  = [emb.vocab[e1].index, emb.vocab[e2].index]
            remb, _ = equalize_boluk(emb.vectors[word2index,:], P)
            emb.vectors[word2index,:] = remb
    #
    sub_mask = [ k for k in mask if k in all_words]
    w2ind_all = [emb.vocab[k].index for k in sub_mask]
    try:
        emb_debiased = MockModel.from_matrix(sub_mask, emb.vectors[w2ind_all, :])
    except:
        import pdb; pdb.set_trace()
    return emb_debiased


if __name__ == '__main__':
    from WEAT.weat_list import WEATLists
    from L101_utils.data_paths import data
    from os.path import join

    n_components = 1

    out_file = join(data, f"glove_bolukbasi_complete.bin")
    mask = list(set([w.lower() for w in WEATLists.weat_vocab]))
    # with open(bolu_definitional_pairs, "r") as f:
    #     p = json.load(f)
    #     mask += list(set([x for x,y in p] + [y for x,y in p]))
    # mask = list(set(mask))
    emb = MockModel.from_file(glove, mock=False)
    emb = hard_debiase(emb, mask=mask, n_components=n_components)
    try:
        emb.save_word2vec_format(out_file, binary=True)
        print(f"saved {out_file}")
    except:
        import traceback
        traceback.print_exc()
        import pdb; pdb.set_trace()
