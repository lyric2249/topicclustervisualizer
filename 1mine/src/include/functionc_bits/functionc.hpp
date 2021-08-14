// [[Rcpp::depends(RcppArmadillo)]]

// #include <iostream>

#include <armadillo>

// #include <pybind11/pybind11.h>
// #include <pybind11/stl.h>
// #include <carma>

// #include <pybind11/numpy.h>
// namespace py = pybind11;

//using namespace std;
//using namespace arma;


namespace funcc
{

class classfuncc
{
    private:

        arma::mat data;

        int ndim;
        int niter, nburn, nthin, nprint;

        double jump_beta, jump_theta, jump_gamma, jump_z, jump_w;
        
        double pr_mean_beta, 
        pr_sd_beta, pr_a_th_sigma, 
        pr_b_th_sigma, pr_mean_theta, 
        pr_a_sigma, pr_b_sigma, 
        pr_mean_gamma, pr_sd_gamma;
        
        double missing;
    
    public:
        void onepl_lsrm_cont_missing(arma::mat data,

        const int ndim,
        const int niter,
        const int nburn,
        const int nthin,
        const int nprint,
        
        const double jump_beta,
        const double jump_theta,
        const double jump_gamma,
        const double jump_z,
        const double jump_w,
        
        const double pr_mean_beta,
        const double pr_sd_beta,
        const double pr_a_th_sigma,
        const double pr_b_th_sigma,
        const double pr_mean_theta,
        const double pr_a_sigma,
        const double pr_b_sigma,
        const double pr_mean_gamma,
        const double pr_sd_gamma,
        const double missing
        );
};

}