from sklearn.decomposition import PCA, KernelPCA


class myKernelPCA(KernelPCA):
    def __init__(self, *args, **kwargs):
        super(myKernelPCA, self).__init__(*args, **kwargs)

    def fit(self, X, y=None):
        self.X_ = X
        super().fit(X, y)

    #  def  β_k(self, X, k):
    def corrected_dot_prod(self, X, Y, center=True):
        α = self.alphas_

        KX_ = self._get_kernel(X, self.X_)
        if center: KX_ = KernelCenterer().fit_transform(KX_)

        KY_ = self._get_kernel(Y, self.X_)
        if center: KY_ = KernelCenterer().fit_transform(KY_)

        KXY = self._get_kernel(X, Y)
        if center: KXY = KernelCenterer().fit_transform(KXY)
        KXY = np.diag(KXY)

        βX = KX_.dot(α)
        βY = KY_.dot(α)

        correction = (βX * βY).sum(axis=1)

        return KXY - correction
