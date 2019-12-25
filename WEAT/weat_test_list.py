from WEAT.weat_slow import WEATTest
from WEAT.weat_list import WEATLists
from L101_utils.data_paths import (bolu_googlew2v, googlew2v,
                                   my_linear_debias, my_linear_debias_k_2,
                                   my_kpca_debias_k_1, data, model, small_googlew2v)
import numpy as np
import WEAT.weat as weat
from L101_utils.mock_model import MockModel
from os.path import join
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.externals import joblib
from L101_src.PCA_sim import corrected_cosine_pca


def w_test(vec_path=None, similarity=cosine_similarity, norm=True):
    emb = MockModel.from_file(vec_path, mock=False)

    if norm:
        emb.vectors /= np.linalg.norm(emb.vectors,  axis=1)[..., None]

    for i, (X,Y,A,B) in enumerate(WEATLists.TEST_LIST):

        if "female" not in WEATLists.INDEX[i].lower(): continue
        print("")
        print(WEATLists.INDEX[i])
        print('WEAT d = ', weat.weat_effect_size(X, Y, A, B, emb, similarity=similarity))
        print('WEAT p = ', weat.weat_p_value(X, Y, A, B, emb, 1000, similarity=similarity))


if __name__ == '__main__':
    sk_model = joblib.load(join(model, "joblib_kpca_cosine_model_k_1.pkl"))
    print((sk_model.lambdas_.flatten() != 0).sum())
    corrected_cosine = lambda X ,Y: sk_model.corrected_cosine_similarity(X, Y)

    sanity_check = False
    if sanity_check:
        sk_model_check = joblib.load(join(model, "joblib_pca_lin_model_k_1.pkl"))
        print(f"pca mean: {sk_model_check.mean_.max()} {sk_model_check.mean_.min()}")
        corrected_cosine = lambda X ,Y: corrected_cosine_pca(sk_model_check, X, Y)
    print("Lodaded model")
    # word_vectors = join(data, "my_weat_mykpca_debias_rbf_vectors_k_1.bin")
    w_test(small_googlew2v, similarity=corrected_cosine)
