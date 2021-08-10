#onepl_lsrm_cont_missing

#def onepl_lsrm_cont_missing()

result = onepl_lsrm_cont_missing(
    data,

    ndim = 2,
    niter = 55000,
    nburn = 5000,
    nthin = 5,
    nprint = 5000,

    jump_beta = 0.3,
    jump_theta = 1.0,
    jump_w = 0.06,
    jump_z = 0.50,
    jump_gamma = 0.01,

    pr_mean_beta = 0,
    pr_sd_beta = 1,
    pr_mean_theta = 0,
    pr_sd_theta = 1,
    pr_mean_gamma = 0.0,
    pr_sd_gamma = 1.0,
    pr_a_sigma = 0.001,
    pr_b_sigma = 0.001,
    pr_a_th_sigma = 0.001,
    pr_b_th_sigma = 0.001,

    missing = 99
)