import numpy as np
import scipy

def procrustes_mine(X, X_star):
    n = X.shape[0]
    m = X.shape[1]

    J = np.identity(n)

    C = X_star.transpose() @ J @ X
    svd_out = np.linalg.svd(C)
    R = svd_out[3] @ svd_out[1].transpose
    s = 1

    tt = np.zeros((m, 1))

    X.new = s * X @ R + tt.reshape(X.shape[0], X.shape[1])
    return X.new

    
def affinityMatrix(diff, K=20, sigma=0.5):
    eps = np.finfo("double").eps #.Machine$double.eps

    N = diff.shape[1]
    diff = (diff + diff.transpose) / 2
    np.fill_diagonal(diff, 0)
    sortedColumns = np.apply_along_axis(sorted, 0, diff).transpose()

    # sortedColumns = as.matrix(t(apply(diff, 2, sort)))
    def finiteMean(x):
        return np.mean(x[np.isfinite(x)])
        # return(mean(x[is.finite(x)]))

    means = np.apply_along_axis(finiteMean, 1, sortedColumns[:, 1:K + 1]) + eps
    # means = apply(sortedColumns[, 1:K + 1], 1, finiteMean) + .Machine$double.eps
    # def avg(x, y):
    #    return (x + y)/2
    Sig = np.mean.outer(means, means) / 3 * 2 + diff / 3 + eps
    Sig[Sig <= eps] = eps

    densities = scipy.stats.norm.pdf(diff, 0, sigma * Sig)
    # densities = dnorm(diff, 0, sigma * Sig, log = FALSE)

    W = (densities + densities.transpose) / 2
    return (W)








