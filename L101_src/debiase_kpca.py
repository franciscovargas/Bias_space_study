import json
import numpy as np
from L101_utils.data_paths import (wikift, bolu_gender_specific,
                                   bolu_equalize_pairs, googlew2v,
                                   bolu_definitional_pairs)
import numpy.linalg as la
from L101_utils.mock_model import MockModel
# from sklearn.decomposition import PCA, KernelPCA
from L101_src.kernel_PCA import myKernelPCA


def neutralise_kpca(X, P):
    X_k = P.transform(X)
    out = X - P.inverse_transform(X_k)
    return out


def get_kpc_projection(X, k=1, mean_rev=True):
    kpca2 = myKernelPCA(kernel="rbf", fit_inverse_transform=True,
                        n_components=k, degree=3)

    kpca2.fit(X)
    return kpca2


def generate_subspace_projection(emb, def_pair_file, n_components):
    with open(def_pair_file, "r") as f:
        pairs = json.load(f)

    matrix = []
    for a, b in pairs:
        center = (emb.vectors[emb.vocab[a.lower()].index] + emb.vectors[emb.vocab[b.lower()].index])/2
        matrix.append(emb.vectors[emb.vocab[a.lower()].index] - center)
        matrix.append(emb.vectors[emb.vocab[b.lower()].index] - center)

    matrix = np.asarray(matrix)
    P = get_kpc_projection(matrix, k=n_components)

    return P


def hard_debiase(emb,
                 gender_specific_file=bolu_gender_specific,
                 equalize_pair_file=bolu_equalize_pairs,
                 def_pair_file=bolu_definitional_pairs,
                 n_components=1,
                 norm=True,
                 mask=None):

    if norm:
        emb.vectors /= np.linalg.norm(emb.vectors,  axis=1)[..., None]

    P = generate_subspace_projection(emb, def_pair_file, n_components)

    with open(gender_specific_file, "r") as f:
        gendered_words = set(json.load(f))

    all_words = set(emb.vocab.keys())
    if mask is None: mask = all_words

    neutral_words = list(set(mask) & all_words)

    word2index = [emb.vocab[k].index for k in neutral_words]

    neutral = emb.vectors[word2index,:]
    emb.vectors[word2index,:] = neutralise_kpca(neutral, P)

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

    out_file = join(data, f"my_weat_mykpca_debias_rbf_vectors_k_{n_components}.bin")
    mask = list(set([w.lower() for w in WEATLists.weat_vocab]))
    emb = MockModel.from_file(googlew2v, mock=False)
    emb = hard_debiase(emb, mask=mask, n_components=n_components)
    try:
        emb.save_word2vec_format(out_file, binary=True)
    except:
        import traceback
        traceback.print_exc()
        import pdb; pdb.set_trace()
