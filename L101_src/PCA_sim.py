import numpy as np


def corrected_dot_prod_pca(pca, X, Y):
    nx, d = X.shape
    ny, d = Y.shape
    I = np.eye(d)

    V = pca.components_[0].reshape(d, 1)
    P = V.dot(V.T)
    C = (I - P)

    Xc, Yc = X.dot(C), Y.dot(C)

    return Xc.dot(Yc.T)


def corrected_cosine_pca(pca, X, Y):
    norm_X = np.sqrt(np.diag(corrected_dot_prod_pca(pca, X, X)))
    norm_Y = np.sqrt(np.diag(corrected_dot_prod_pca(pca, Y, Y)))
    norm_XY = norm_X.reshape(-1,1).dot(norm_Y.reshape(1,-1))

    XY = corrected_dot_prod_pca(pca, X, Y)

    out = XY / ( norm_XY )
    return out
