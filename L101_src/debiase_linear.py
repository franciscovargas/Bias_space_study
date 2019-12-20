import json
import numpy as np


def get_pc_projection_boluk(X, k=1, mean_rev=True):
    mean_rev = int(mean_rev)
    n, d = X.shape
    X = X - mean_rev * X.mean(axis=0)
    C = (X.T.dot(X) / n)
    D, V = la.eigh(C)
    V = V[:, :k]
    return V.dot(V.T), V


def create_nu_boluk(X, P):
    mu = X.mean(axis=0)
    nu = mu - P.dot(mu)
    return nu, P.dot(mu)


def equalize_boluk(E, P, N=None, debug=True):

    nu = (np.eye(len(P)) - P).dot( E.mean(axis=0) )
    E = (E - E.mean(axis=0)[None, ...]).dot(P)

    E /= np.linalg.norm(E, axis=1)[..., None]

    v = np.linalg.norm(nu)
    fac = np.sqrt(1 - v**2)
    remb = nu +  fac * E

    return remb, E


def neutralise_boluk(X, P):
    print("start neutralise")
    I = np.eye(P.shape[0])
    out =  X.dot( (I - P).T )
    print("done matrix mult (neutralise)", out.shape)
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
                 norm=True):

#     emb = deepcopy(emb)
    if norm:
        emb.vectors /= np.linalg.norm(emb.vectors,  axis=1)[..., None]

    P, V = generate_subspace_projection(emb, def_pair_file, n_components)
    assert (P == P.T).all()

    with open(gender_specific_file, "r") as f:
        gendered_words = set(json.load(f))

    all_words = set(emb.vocab.keys())
    neutral_words = all_words - gendered_words

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
        if (e1 in all_words and e2 in all_words):
            word2index  = [emb.vocab[e1].index, emb.vocab[e2].index]
            remb, _ = equalize_boluk(emb.vectors[word2index,:], P)
            emb.vectors[word2index,:] = remb
    return emb


if __name__ == '__main__':
    emb = MockModel.from_file(googlew2v, mock=False)
    embnasius = hard_debiase(emb)
