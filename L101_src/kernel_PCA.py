from sklearn.decomposition import PCA, KernelPCA
from sklearn.preprocessing import KernelCenterer
import numpy as np


class myKernelPCA(KernelPCA):
    def __init__(self, *args, **kwargs):
        super(myKernelPCA, self).__init__(*args, **kwargs)

    def fit(self, X, y=None):
        self.X_ = X
        super().fit(X, y)

    #  def  β_k(self, X, k):
    def corrected_dot_prod(self, X, Y, center=True):

        non_zeros = np.flatnonzero(self.lambdas_)
        α  = np.zeros_like(self.alphas_)
        α [:, non_zeros] = (self.alphas_[:, non_zeros]
                            / np.sqrt(self.lambdas_[non_zeros]))

        # self._centerer.transform(self._get_kernel(X, self.X_fit_))
        KX_ = self._get_kernel(X, self.X_fit_)
        if center: KX_ = self._centerer.transform(KX_)

        KY_ = self._get_kernel(Y, self.X_fit_)
        if center: KY_ = self._centerer.transform(KY_)

        KXY = self._get_kernel(X, Y)
        # if center: KXY = KernelCenterer().fit_transform(KXY)
        # KXY = np.diag(KXY)

        βX = KX_.dot(α)
        βY = KY_.dot(α)

        correction = (βX.dot(βY.T))
        # import pdb; pdb.set_trace()
        # kkkkk
        return KXY - correction

    def corrected_cosine_similarity(self, X, Y, center=False):
        norm_X = np.sqrt(np.diag(self.corrected_dot_prod(X, X, center=center)))
        norm_Y = np.sqrt(np.diag(self.corrected_dot_prod(Y, Y, center=center)))
        norm_XY = norm_X.reshape(-1,1).dot(norm_Y.reshape(1,-1))

        XY = self.corrected_dot_prod(X, Y, center=center)

        out = XY / ( norm_XY )
        return out
