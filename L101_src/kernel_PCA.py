from sklearn.decomposition import PCA, KernelPCA
from sklearn.preprocessing import KernelCenterer
import numpy as np


class myKernelPCA(KernelPCA):
    def __init__(self, *args, **kwargs):
        super(myKernelPCA, self).__init__(*args, **kwargs)

    def fit(self, X, y=None):
        self.X_ = X
        super().fit(X, y)

    def kyx_center(self, Kxy, KxX, KyX):
        K = self._get_kernel(self.X_fit_)

        Kxy_c = Kxy.copy()

        Kxy_c -= KxX.mean(axis=1)[..., None]
        Kxy_c -= KyX.mean(axis=1)[None, ...]

        Kxy_c += K.mean()

        return Kxy_c

    #  def  β_k(self, X, k):
    def corrected_dot_prod(self, X, Y, center=True):

        # non_zeros = np.flatnonzero(self.lambdas_)
        # import pdb; pdb.set_trace()
        λ_non_zeros = np.flatnonzero(self.lambdas_)
        α = np.zeros_like(self.alphas_)
        α[:, λ_non_zeros] = (self.alphas_[:, λ_non_zeros]
                                       / np.sqrt(self.lambdas_[λ_non_zeros]))
        # import pdb; pdb.set_trace()

        KX_ = self._get_kernel(X, self.X_fit_)
        KXunc = KX_.copy()
        if center: KX_ = self._centerer.transform(KX_)

        KY_ = self._get_kernel(Y, self.X_fit_)
        KYunc = KY_.copy()
        if center: KY_ = self._centerer.transform(KY_)

        KXY = self._get_kernel(X, Y)
        if center: KXY = self.kyx_center(KXY, KXunc, KYunc)

        βX = KX_.dot(α)
        βY = KY_.dot(α)

        correction = (βX.dot(βY.T))

        return KXY - correction

    def corrected_cosine_similarity(self, X, Y, center=True):
        norm_X = np.sqrt(np.diag(self.corrected_dot_prod(X, X, center=center)))
        norm_Y = np.sqrt(np.diag(self.corrected_dot_prod(Y, Y, center=center)))
        norm_XY = norm_X.reshape(-1,1).dot(norm_Y.reshape(1,-1))

        XY = self.corrected_dot_prod(X, Y, center=center)

        out = XY / ( norm_XY )
        return out
