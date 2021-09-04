import numpy as np
import scipy

def procrustes_mine(X, X_star):
    
    n, m = X.shape
    J = np.identity(n)

    C = X_star.transpose() @ J @ X

    svd_out = np.linalg.svd(C)

    R = svd_out[2] @ svd_out[0].transpose()
    s = 1

    tt = np.zeros((m, 1))

    X_new = s * X @ R + np.full((n, m), tt.transpose())
    return X_new





def affinityMatrix(diff, K=20, sigma=0.5):
    eps = np.finfo("double").eps #.Machine$double.eps

    N = diff.shape[1] #TODO: For what?
    diff = (diff + diff.transpose()) / 2
    np.fill_diagonal(diff, 0)
    sortedColumns = np.apply_along_axis(sorted, 0, diff).transpose()

    # sortedColumns = as.matrix(t(apply(diff, 2, sort)))
    def finiteMean(x):
        return np.mean(x[np.isfinite(x)])
        # return(mean(x[is.finite(x)]))

    means = np.apply_along_axis(finiteMean, 1, sortedColumns[:, 1:(K + 1)]) + eps
    # means = apply(sortedColumns[, 1:K + 1], 1, finiteMean) + .Machine$double.eps
    # def avg(x, y):
    #    return (x + y)/2
    Sig = np.outer(means, means) / 3 * 2 + (diff / 3) + (eps)
    Sig[Sig <= eps] = eps

    densities = scipy.stats.norm.pdf(diff, 0, sigma * Sig)
    # densities = dnorm(diff, 0, sigma * Sig, log = FALSE)

    W = (densities + densities.transpose()) / 2
    return (W)






def MCMC_process(data, niter, nburn, nthin, ndim, output):

    import time

    start = time.time()

    nsample, nitem = data.shape
    nmcmc = int((niter - nburn) / nthin)

    max_address = np.argmax(output['map'])
    
    w_star = output['w'][max_address, :, :]
    z_star = output['z'][max_address, :, :]
    
    w_proc = np.zeros((nmcmc, nitem, ndim), )
    z_proc = np.zeros((nmcmc, nsample, ndim), )

    for iter in range(nmcmc):
        z_iter = output['z'][iter, :, :]

        if iter != max_address:
            z_proc[iter, :, :] = procrustes_mine(z_iter, z_star)
        else:
            z_proc[iter, :, :] = z_iter

        w_iter = output['w'][iter, :, :]

        if iter != max_address:
            w_proc[iter, :, :] = procrustes_mine(w_iter, w_star) # TODO: ======= 210717 =======
        else:
            w_proc[iter, :, :] = w_iter

    w_est = np.empty((nitem, ndim))

    for i in range(nitem):
        for j in range(ndim):
            w_est[i, j] = w_proc[:, i, j].mean()

    z_est = np.empty((nsample, ndim,))

    for k in range(nsample):
        for j in range(ndim):
            z_est[k, j] = z_proc[:, k, j].mean()

    beta_est = output["beta"].mean()
    theta_est = output["theta"].mean()

    # beta_est = apply(output["beta"], 2, mean)
    # theta_est = apply(output["theta"], 2, mean)

    sigma_theta_est = output["sigma_theta"].mean()
    gamma_est = output["gamma"].mean()

    output_new = {"beta_estimate": beta_est,
                "theta_estimate": theta_est,
                "sigma_theta_estimate": sigma_theta_est,
                "gamma_estimate": gamma_est,
                "z_estimate": z_est,
                "w_estimate": w_est,
                "beta": output["beta"],
                "theta": output["theta"],
                "theta_sd": output["sigma_theta"],
                "gamma": output["gamma"],
                "z": z_proc,
                "w": w_proc,
                "accept_beta": output["accept_beta"],
                "accept_theta": output["accept_theta"],
                "accept_w": output["accept_w"],
                "accept_z": output["accept_z"],
                "accept_gamma": output["accept_gamma"]
                }

    print(round(time.time() - start, 2), " seconds")
    
    return output_new
