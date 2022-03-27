// #include <iostream>

#include <string>
#include <armadillo>

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>

// pybind11::array_t<double> Mat;
// #include <carma>

namespace py = pybind11;

//using namespace std;
//using namespace arma;


//#ifndef FUNCTIONC_H
//#define FUNCTIONC_H

arma::mat _data;

int _ndim;
int _niter, _nburn, _nthin, _nprint;

double _jump_beta, _jump_theta, _jump_gamma, _jump_z, _jump_w;

double _pr_mean_beta,
_pr_sd_beta, _pr_a_th_sigma,
_pr_b_th_sigma, _pr_mean_theta,
_pr_a_sigma, _pr_b_sigma,
_pr_mean_gamma, _pr_sd_gamma;

double _missing;



//Constructor
std::vector<py::array_t<double>> 
onepl_lsrm_cont_missing
(
    arma::Mat<double> input,

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

//#endif