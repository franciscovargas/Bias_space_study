from sklearn.decomposition import PCA, KernelPCA
from sklearn.preprocessing import KernelCenterer
from sklearn.utils.validation import (check_is_fitted, check_array)
import numpy as np
from sklearn.metrics.pairwise import _parallel_pairwise
from scipy import linalg



class myKernelPCA(KernelPCA):
    def __init__(self, *args, **kwargs):
        super(myKernelPCA, self).__init__(*args, **kwargs)

    def _get_kernel(self, X, Y=None):
        """
        Modified to handle vectorised (only) custom kernels
        """
        if isinstance(self.kernel, str):
            return super(myKernelPCA, self)._get_kernel(X, Y)
        elif callable(self.kernel):
            return _parallel_pairwise(X, Y, self.kernel, None, **{})
        else:
            raise ValueError("Neither a string nor a callable provided")

    def _get_kernel_train(self, X, Y):
        """
        the kernel at train time which substraces phi(male) - phi(female)
        pairs in the RKHS via the kernel trick. For kernel at test time
        use self._get_kernel
        """

        KX = self._get_kernel(X)
        KY = self._get_kernel(Y)
        KXY = self._get_kernel(X, Y)

        return KX - KXY - KXY.T + KY

    def fit(self, X, Y):
        """Fit the model from data in X.
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training vector corresponding to female related word vectors
            in defnition sets, where n_samples in the number of samples
            and n_features is the number of features.

        Y : array-like, shape (n_samples, n_features)
            male counterpart of X.
        Returns
        -------
        self : object
            Returns the instance itself.
        """
        X = check_array(X, accept_sparse='csr', copy=self.copy_X)
        Y = check_array(Y, accept_sparse='csr', copy=self.copy_X)

        self._centerer = KernelCenterer()
        K = self._get_kernel_train(X, Y)
        self._fit_transform(K)

        self.X_fit_ = X
        self.Y_fit_ = Y
        return self

    def kyx_center(self, Kxy, KxX, KyX):
        K = self._get_kernel_train(self.X_fit_, self.Y_fit_)

        Kxy_c = Kxy.copy()

        Kxy_c -= KxX.mean(axis=1)[..., None]
        Kxy_c -= KyX.mean(axis=1)[None, ...]

        Kxy_c += K.mean()

        return Kxy_c

    def transform(self, X):
        """Transform X. Adapted to the paired bias training set (definitiion sets)
        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
        Returns
        -------
        X_new : array-like, shape (n_samples, n_components)
        """
        check_is_fitted(self)

        # Compute centered gram matrix between X and training data X_fit_, Y_fit_
        K = self._get_kernel(X, self.X_fit_)
        K -= self._get_kernel(X, self.Y_fit_)
        K = self._centerer.transform(K)

        # scale eigenvectors (properly account for null-space for dot product)
        non_zeros = np.flatnonzero(self.lambdas_)
        scaled_alphas = np.zeros_like(self.alphas_)
        scaled_alphas[:, non_zeros] = (self.alphas_[:, non_zeros]
                                       / np.sqrt(self.lambdas_[non_zeros]))

        # Project with a scalar product between K and the scaled eigenvectors
        return np.dot(K, scaled_alphas)

    def corrected_dot_prod(self, X, Y, center=True):

        λ_non_zeros = np.flatnonzero(self.lambdas_)
        α = np.zeros_like(self.alphas_)
        α[:, λ_non_zeros] = (self.alphas_[:, λ_non_zeros]
                                       / np.sqrt(self.lambdas_[λ_non_zeros]))

        KX_ = self._get_kernel(X, self.X_fit_)
        KX_ -= self._get_kernel(X, self.Y_fit_)
        KXunc = KX_.copy()
        if center: KX_ = self._centerer.transform(KX_)

        KY_ = self._get_kernel(Y, self.X_fit_)
        KY_ -= self._get_kernel(Y, self.Y_fit_)
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

    def fit_my_inverse_transform(self, X):
        """
        Ideally can be fitted on set larger than the original 10 point
        training set from Bolukbasi et al. Does not project X into subspace
        to learn the kernel directly instead learns (x, phi(x)) pairs.
        """
        if hasattr(X, "tocsr"):
            raise NotImplementedError("Inverse transform not implemented for "
                                      "sparse matrices!")

        n_samples = X.shape[0]
        K = self._get_kernel(X)
        K.flat[::n_samples + 1] += self.alpha
        self.dual_coef_ = linalg.solve(K, X, sym_pos=True, overwrite_a=True)
        self.X_transformed_fit_ = X
        self.fit_inverse_transform = True
