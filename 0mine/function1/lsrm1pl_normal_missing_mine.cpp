// [[Rcpp::depends(RcppArmadillo)]]

#include <iostream>
#include <armadillo>
#include <carma/carma.h>

#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/pybind11.h>




using namespace std;
using namespace arma;

// Armadillo documentation is available at:
// http://arma.sourceforge.net/docs.html

// NOTE: the C++11 "auto" keyword is not recommended for use with Armadillo objects and functions



field<cube> onepl_lsrm_cont_missing
(
    mat data,

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













int main(int argc, char** argv) 
{
    cout << "Armadillo version: " << arma_version::as_string() << endl;

    // construct a matrix according to given size and form of element initialisation
    mat A(2, 3, fill::zeros);

    // .n_rows and .n_cols are read only
    cout << "A.n_rows: " << A.n_rows << endl;
    cout << "A.n_cols: " << A.n_cols << endl;

    A(1, 2) = 456.0;  // access an element (indexing starts at 0)
    A.print("A:");

    A = 5.0;         // scalars are treated as a 1x1 matrix
    A.print("A:");

    A.set_size(4, 5); // change the size (data is not preserved)

    A.fill(5.0);     // set all elements to a specific value
    A.print("A:");

    A = { { 0.165300, 0.454037, 0.995795, 0.124098, 0.047084 },
          { 0.688782, 0.036549, 0.552848, 0.937664, 0.866401 },
          { 0.348740, 0.479388, 0.506228, 0.145673, 0.491547 },
          { 0.148678, 0.682258, 0.571154, 0.874724, 0.444632 },
          { 0.245726, 0.595218, 0.409327, 0.367827, 0.385736 } };

    A.print("A:");

    // determinant
    cout << "det(A): " << det(A) << endl;

    // inverse
    cout << "inv(A): " << endl << inv(A) << endl;

    // save matrix as a text file
    A.save("A.txt", raw_ascii);

    // load from file
    mat B;
    B.load("A.txt");

    // submatrices
    cout << "B( span(0,2), span(3,4) ):" << endl << B(span(0, 2), span(3, 4)) << endl;

    cout << "B( 0,3, size(3,2) ):" << endl << B(0, 3, size(3, 2)) << endl;

    cout << "B.row(0): " << endl << B.row(0) << endl;

    cout << "B.col(1): " << endl << B.col(1) << endl;

    // transpose
    cout << "B.t(): " << endl << B.t() << endl;

    // maximum from each column (traverse along rows)
    cout << "max(B): " << endl << max(B) << endl;

    // maximum from each row (traverse along columns)
    cout << "max(B,1): " << endl << max(B, 1) << endl;

    // maximum value in B
    cout << "max(max(B)) = " << max(max(B)) << endl;

    // sum of each column (traverse along rows)
    cout << "sum(B): " << endl << sum(B) << endl;

    // sum of each row (traverse along columns)
    cout << "sum(B,1) =" << endl << sum(B, 1) << endl;

    // sum of all elements
    cout << "accu(B): " << accu(B) << endl;

    // trace = sum along diagonal
    cout << "trace(B): " << trace(B) << endl;

    // generate the identity matrix
    mat C = eye<mat>(4, 4);

    // random matrix with values uniformly distributed in the [0,1] interval
    mat D = randu<mat>(4, 4);
    D.print("D:");

    // row vectors are treated like a matrix with one row
    rowvec r = { 0.59119, 0.77321, 0.60275, 0.35887, 0.51683 };
    r.print("r:");

    // column vectors are treated like a matrix with one column
    vec q = { 0.14333, 0.59478, 0.14481, 0.58558, 0.60809 };
    q.print("q:");

    // convert matrix to vector; data in matrices is stored column-by-column
    vec v = vectorise(A);
    v.print("v:");

    // dot or inner product
    cout << "as_scalar(r*q): " << as_scalar(r * q) << endl;

    // outer product
    cout << "q*r: " << endl << q * r << endl;

    // multiply-and-accumulate operation (no temporary matrices are created)
    cout << "accu(A % B) = " << accu(A % B) << endl;

    // example of a compound operation
    B += 2.0 * A.t();
    B.print("B:");

    // imat specifies an integer matrix
    imat AA = { { 1, 2, 3 },
                { 4, 5, 6 },
                { 7, 8, 9 } };

    imat BB = { { 3, 2, 1 },
                { 6, 5, 4 },
                { 9, 8, 7 } };

    // comparison of matrices (element-wise); output of a relational operator is a umat
    umat ZZ = (AA >= BB);
    ZZ.print("ZZ:");

    // cubes ("3D matrices")
    cube Q(B.n_rows, B.n_cols, 2);

    Q.slice(0) = B;
    Q.slice(1) = 2.0 * B;

    Q.print("Q:");

    // 2D field of matrices; 3D fields are also supported
    field<mat> output(4, 3);

    for (uword col = 0; col < output.n_cols; ++col)
        for (uword row = 0; row < output.n_rows; ++row)
        {
            output(row, col) = randu<mat>(2, 3);  // each element in field<mat> is a matrix
        }

    output.print("output:");

    return 0;
}






field<cube> onepl_lsrm_cont_missing
(
    mat data,

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
)
{
    const int nsample = data.n_rows;
    const int nitem = data.n_cols;

    int i, j, k, count, accept;
    double num, den, old_like_beta, new_like_beta, old_like_theta, new_like_theta;
    double old_like_z, new_like_z, old_like_w, new_like_w, old_like_gamma, new_like_gamma;
    double ratio, un, dist_temp, dist_old_temp, dist_new_temp;
    double post_a_sigma, post_b_sigma, post_a_th_sigma, post_b_th_sigma;
    double pr_mean_z = 0.0, pr_sd_z = 1.0, pr_mean_w = 0.0, pr_sd_w = 1.0, pr_sd = 1.0, pr_sd_theta = 1.0, mle;


    dvec oldbeta(nitem, fill::randu);
    oldbeta = oldbeta * 4.0 - 2.0;
    dvec newbeta = oldbeta;

    dvec oldtheta(nsample, fill::randu);
    oldtheta = oldtheta * 4.0 - 2.0;
    dvec newtheta = oldtheta;

    dmat oldz(nsample, ndim, fill::randu);
    oldz = oldz * 2.0 - 1.0;
    dmat newz = oldz;

    dmat oldw(nitem, ndim, fill::randu);
    oldw = oldw * 2.0 - 1.0;
    dmat neww = oldw;

    double oldgamma = 1, newgamma = 1; //gamma = log(gamma)
    dmat samp_beta((niter - nburn) / nthin, nitem, fill::zeros);
    dmat samp_theta((niter - nburn) / nthin, nsample, fill::zeros);
    dcube samp_z(((niter - nburn) / nthin), nsample, ndim, fill::zeros);
    dcube samp_w(((niter - nburn) / nthin), nitem, ndim, fill::zeros);
    dvec samp_sd_theta((niter - nburn) / nthin, fill::zeros);
    dvec samp_sd((niter - nburn) / nthin, fill::zeros);
    dvec samp_mle((niter - nburn) / nthin, fill::zeros);
    dvec samp_gamma(((niter - nburn) / nthin), fill::zeros);

    dvec accept_beta(nitem, fill::zeros);
    dvec accept_theta(nsample, fill::zeros);
    dvec accept_z(nsample, fill::zeros);
    dvec accept_w(nitem, fill::zeros);
    double accept_gamma = 0;

    accept = count = 0;

    dmat dist(nsample, nitem, fill::zeros);
    dvec old_dist_k(nitem, fill::zeros);
    dvec new_dist_k(nitem, fill::zeros);
    dvec old_dist_i(nsample, fill::zeros);
    dvec new_dist_i(nsample, fill::zeros);

    for (int iter = 0; iter < niter; iter++) {

        //dist(j,i) is distance of z_j and w_i
        dist.fill(0.0);
        for (i = 0; i < nitem; i++) {
            for (k = 0; k < nsample; k++) {
                dist_temp = 0.0;
                for (j = 0; j < ndim; j++) dist_temp += pow((oldz(k, j) - oldw(i, j)), 2.0);
                dist(k, i) = sqrt(dist_temp);
            }
        }

        // beta update
        for (i = 0; i < nitem; i++) {
            newbeta(i) = oldbeta(i) + jump_beta * randn();
            // newbeta(i) = R::rnorm(oldbeta(i), jump_beta);
            old_like_beta = new_like_beta = 0.0;
            for (k = 0; k < nsample; k++) {
                if (data(k, i) != missing) {
                    new_like_beta += -pow((data(k, i) - newbeta(i) - oldtheta(k) + oldgamma * dist(k, i)), 2) / (2 * pow(pr_sd, 2));
                    old_like_beta += -pow((data(k, i) - oldbeta(i) - oldtheta(k) + oldgamma * dist(k, i)), 2) / (2 * pow(pr_sd, 2));
                }
            }

            num = new_like_beta + log_normpdf(newbeta(i), pr_mean_beta, pr_sd_beta);
            den = old_like_beta + log_normpdf(oldbeta(i), pr_mean_beta, pr_sd_beta);
            ratio = num - den;

            if (ratio > 0.0) accept = 1;
            else {
                un = randu();
                if (log(un) < ratio) accept = 1;
                else accept = 0;
            }

            if (accept == 1) {
                oldbeta(i) = newbeta(i);
                accept_beta(i) += 1.0 / (niter * 1.0);
            }
            else newbeta(i) = oldbeta(i);

        }

        // theta update
        for (k = 0; k < nsample; k++) {
            newtheta(k) = oldtheta(k) + jump_theta * randn();
            old_like_theta = new_like_theta = 0.0;

            for (i = 0; i < nitem; i++) {
                if (data(k, i) != missing) {
                    new_like_theta += -pow((data(k, i) - oldbeta(i) - newtheta(k) + oldgamma * dist(k, i)), 2) / (2 * pow(pr_sd, 2));
                    old_like_theta += -pow((data(k, i) - oldbeta(i) - oldtheta(k) + oldgamma * dist(k, i)), 2) / (2 * pow(pr_sd, 2));
                }
            }
            num = new_like_theta + log_normpdf(newtheta(k), pr_mean_theta, pr_sd_theta);
            den = old_like_theta + log_normpdf(oldtheta(k), pr_mean_theta, pr_sd_theta);
            ratio = num - den;

            if (ratio > 0.0) accept = 1;
            else {
                un = randu();
                if (log(un) < ratio) accept = 1;
                else accept = 0;
            }

            if (accept == 1) {
                oldtheta(k) = newtheta(k);
                accept_theta(k) += 1.0 / (niter * 1.0);
            }
            else newtheta(k) = oldtheta(k);
        }

        // gamma(log(gamma)) update
        newgamma = exp(log(oldgamma) + jump_gamma * randn());
        old_like_gamma = 0.0;
        new_like_gamma = 0.0;

        for (k = 0; k < nsample; k++) {
            for (i = 0; i < nitem; i++) {
                if (data(k, i) != missing) {
                    new_like_gamma += -pow((data(k, i) - oldbeta(i) - newtheta(k) + newgamma * dist(k, i)), 2) / (2 * pow(pr_sd, 2));
                    old_like_gamma += -pow((data(k, i) - oldbeta(i) - newtheta(k) + oldgamma * dist(k, i)), 2) / (2 * pow(pr_sd, 2));
                }
            }
        }


        num = new_like_gamma +
            log_normpdf(log(oldgamma), log(newgamma), jump_gamma) +
            log_normpdf(log(newgamma), pr_mean_gamma, pr_sd_gamma);
        //R::dlnorm(oldgamma, std::log(newgamma), jump_gamma, 1) + 
        //R::dlnorm(newgamma, pr_mean_gamma, pr_sd_gamma, 1);
        den = old_like_gamma +
            log_normpdf(log(newgamma), log(oldgamma), jump_gamma) +
            log_normpdf(log(oldgamma), pr_mean_gamma, pr_sd_gamma);
        //R::dlnorm(newgamma, std::log(oldgamma), jump_gamma, 1) + 
        //R::dlnorm(oldgamma, pr_mean_gamma, pr_sd_gamma, 1);
        ratio = num - den;

        if (ratio > 0.0) accept = 1;
        else {
            un = randu();
            if (log(un) < ratio) accept = 1;
            else accept = 0;
        }

        if (accept == 1) {
            oldgamma = newgamma;
            accept_gamma += 1.0 / (niter * 1.0);
        }
        else newgamma = oldgamma;

        // zj update
        for (k = 0; k < nsample; k++) {
            for (j = 0; j < ndim; j++) newz(k, j) = oldz(k, j) + jump_z * randn();
            old_like_z = new_like_z = 0.0;

            //calculate distance of oldw and newz
            for (i = 0; i < nitem; i++) {
                dist_old_temp = dist_new_temp = 0.0;
                for (j = 0; j < ndim; j++) {
                    dist_new_temp += pow((newz(k, j) - oldw(i, j)), 2.0);
                    dist_old_temp += pow((oldz(k, j) - oldw(i, j)), 2.0);
                }
                new_dist_k(i) = sqrt(dist_new_temp);
                old_dist_k(i) = sqrt(dist_old_temp);
            }

            //calculate likelihood
            for (i = 0; i < nitem; i++) {
                if (data(k, i) != missing) {
                    new_like_z += -pow((data(k, i) - oldbeta(i) - oldtheta(k) + oldgamma * new_dist_k(i)), 2) / (2 * pow(pr_sd, 2));
                    old_like_z += -pow((data(k, i) - oldbeta(i) - oldtheta(k) + oldgamma * old_dist_k(i)), 2) / (2 * pow(pr_sd, 2));
                }
            }

            num = den = 0.0;
            for (j = 0; j < ndim; j++) {
                num += log_normpdf(newz(k, j), pr_mean_z, pr_sd_z);
                //#num += scipy.stats.norm.logpdf(newz[k, j], pr_mean_z, pr_sd_z)
                den += log_normpdf(oldz(k, j), pr_mean_z, pr_sd_z);
            }
            //Rprintf("%.3f %.3f %.3f %.3f\n", num, den, new_like_z, old_like_z);
            //arma::dvec newzz = dmvnorm(newz.cols(2*j,2*j+1),pr_mean_z,pr_cov_z,TRUE);
            //arma::dvec oldzz = dmvnorm(oldz.cols(2*j,2*j+1),pr_mean_z,pr_cov_z,TRUE);

            num += new_like_z;
            den += old_like_z;
            ratio = num - den;

            if (ratio > 0.0) accept = 1;
            else {
                un = randu();
                if (log(un) < ratio) accept = 1;
                else accept = 0;
            }

            if (accept == 1) {
                for (j = 0; j < ndim; j++) oldz(k, j) = newz(k, j);
                accept_z(k) += 1.0 / (niter * 1.0);
            }
            else {
                for (j = 0; j < ndim; j++) newz(k, j) = oldz(k, j);
            }
        }

        // wi update
        for (i = 0; i < nitem; i++) {
            for (j = 0; j < ndim; j++) neww(i, j) = oldw(i, j) + jump_w * randn();
            old_like_w = new_like_w = 0.0;

            //calculate distance of neww and oldz
            for (k = 0; k < nsample; k++) {
                dist_old_temp = dist_new_temp = 0.0;
                for (j = 0; j < ndim; j++) {
                    dist_new_temp += pow((oldz(k, j) - neww(i, j)), 2.0);
                    dist_old_temp += pow((oldz(k, j) - oldw(i, j)), 2.0);
                }
                new_dist_i(k) = sqrt(dist_new_temp);
                old_dist_i(k) = sqrt(dist_old_temp);
            }

            //calculate likelihood
            for (k = 0; k < nsample; k++) {
                if (data(k, i) != missing) {
                    new_like_w += -pow((data(k, i) - oldbeta(i) - oldtheta(k) + oldgamma * new_dist_i(k)), 2) / (2 * pow(pr_sd, 2));
                    old_like_w += -pow((data(k, i) - oldbeta(i) - oldtheta(k) + oldgamma * old_dist_i(k)), 2) / (2 * pow(pr_sd, 2));
                }
            }

            num = den = 0.0;
            for (j = 0; j < ndim; j++) {
                num += log_normpdf(neww(i, j), pr_mean_w, pr_sd_w);
                den += log_normpdf(oldw(i, j), pr_mean_w, pr_sd_w);
            }

            num += new_like_w;
            den += old_like_w;
            ratio = num - den;

            if (ratio > 0.0) accept = 1;
            else {
                un = randu();
                if (log(un) < ratio) accept = 1;
                else accept = 0;
            }

            if (accept == 1) {
                for (j = 0; j < ndim; j++) oldw(i, j) = neww(i, j);
                accept_w(i) += 1.0 / (niter * 1.0);
            }
            else {
                for (j = 0; j < ndim; j++) neww(i, j) = oldw(i, j);
            }
        }


        //sigma_theta update with gibbs
        post_a_th_sigma = 2 * pr_a_th_sigma + nsample;
        post_b_th_sigma = pr_b_th_sigma;
        for (j = 0; j < nsample; j++) post_b_th_sigma += pow((oldtheta(j) - pr_mean_theta), 2.0);
        pr_sd_theta = sqrt(2 * post_b_th_sigma * (1.0 / chi2rnd(post_a_th_sigma)));

        //dist(j,i) is distance of z_j and w_i
        dist.fill(0.0);
        for (i = 0; i < nitem; i++) {
            for (k = 0; k < nsample; k++) {
                dist_temp = 0.0;
                for (j = 0; j < ndim; j++) dist_temp += pow((oldz(k, j) - oldw(i, j)), 2.0);
                dist(k, i) = sqrt(dist_temp);
            }
        }


        //sigma update with gibbs
        post_a_sigma = 2 * pr_a_sigma + nsample * nitem;
        post_b_sigma = pr_b_sigma;
        for (j = 0; j < nsample; j++) {
            for (i = 0; i < nitem; i++) post_b_sigma += pow((data(j, i) - oldbeta(i) - oldtheta(j) + oldgamma * dist(j, i)), 2.0) / 2;
        }
        pr_sd = sqrt(2 * post_b_sigma * (1.0 / chi2rnd(post_a_th_sigma)));

        // burn, thin
        if (iter >= nburn && iter % nthin == 0) {
            for (i = 0; i < nitem; i++) samp_beta(count, i) = oldbeta(i);
            for (k = 0; k < nsample; k++) samp_theta(count, k) = oldtheta(k);
            for (i = 0; i < nitem; i++) {
                for (j = 0; j < ndim; j++) {
                    samp_w(count, i, j) = oldw(i, j);
                }
            }
            for (k = 0; k < nsample; k++) {
                for (j = 0; j < ndim; j++) {
                    samp_z(count, k, j) = oldz(k, j);
                }
            }

            samp_gamma(count) = oldgamma;
            samp_sd_theta(count) = pr_sd_theta;
            samp_sd(count) = pr_sd;

            mle = 0.0;
            for (i = 0; i < nitem; i++) mle += log_normpdf(oldbeta(i), pr_mean_beta, pr_sd_beta);
            for (k = 0; k < nsample; k++) mle += log_normpdf(oldtheta(k), pr_mean_theta, pr_sd_theta);
            for (i = 0; i < nitem; i++)
                for (j = 0; j < ndim; j++) mle += log_normpdf(oldw(i, j), pr_mean_w, pr_sd_w);
            for (k = 0; k < nsample; k++)
                for (j = 0; j < ndim; j++) mle += log_normpdf(oldz(k, j), pr_mean_z, pr_sd_z);
            for (k = 0; k < nsample; k++) {
                for (i = 0; i < nitem; i++) {
                    mle += -pow((data(k, i) - oldbeta(i) - oldtheta(k) + oldgamma * dist(k, i)), 2) / (2 * pow(pr_sd, 2));
                }
            }
            mle += log_normpdf(log(oldgamma), pr_mean_gamma, pr_sd_gamma);
            samp_mle(count) = mle;

            count++;
        }

        if (iter % nprint == 0)
        {
            printf("Iteration: %.5u ", iter);
            for (i = 0; i < nitem; i++)
            {
                printf("% .3f ", oldbeta(i));
            }
            printf(" %.3f ", oldgamma);
            printf(" %.3f\n", pr_sd_theta);
        }

    } //for end





    field<cube> output(1, 13); 
    dcube samp_beta_cube(samp_beta.n_rows, samp_beta.n_cols, 1);
    samp_beta_cube.slice(0) = samp_beta;
    output(0, 0) = samp_beta_cube; //dmat

    dcube samp_theta_cube(samp_theta.n_rows, samp_theta.n_cols, 1);
    samp_theta_cube.slice(0) = samp_theta;
    output(0, 1) = samp_theta_cube; //dmat

    output(0, 2) = samp_z; //dcube
    output(0, 3) = samp_w; //dcube


    dcube samp_gamma_cube(samp_gamma.n_elem, 1, 1);
    samp_gamma_cube.slice(0) = samp_gamma;
    output(0, 4) = samp_gamma_cube; //dvec

    dcube samp_sd_theta_cube(samp_sd_theta.n_elem, 1, 1);
    samp_sd_theta_cube.slice(0) = samp_sd_theta;
    output(0, 5) = samp_sd_theta_cube; //dvec
    
    dcube samp_sd_cube(samp_sd.n_elem, 1, 1);
    samp_sd_cube.slice(0) = samp_sd;
    output(0, 6) = samp_sd_cube; //dvec
    
    dcube samp_mle_cube(samp_mle.n_elem, 1, 1);
    samp_mle_cube.slice(0) = samp_mle;
    output(0, 7) = samp_mle_cube; //dvec
    
    dcube accept_beta_cube(accept_beta.n_elem, 1, 1);
    accept_beta_cube.slice(0) = accept_beta;
    output(0, 8) = accept_beta_cube; //dvec
    
    dcube accept_theta_cube(accept_theta.n_elem, 1, 1);
    accept_theta_cube.slice(0) = accept_theta;
    output(0, 9) = accept_theta_cube; //dvec
    
    dcube accept_z_cube(accept_z.n_elem, 1, 1);
    accept_z_cube.slice(0) = accept_z;
    output(0, 10) = accept_z_cube; //dvec
    
    dcube accept_w_cube(accept_w.n_elem, 1, 1);
    accept_w_cube.slice(0) = accept_w;
    output(0, 11) = accept_w_cube; //dvec
    
    output(0, 12) = accept_gamma; //double
    
    // Rcpp::List output;
    // output["beta"]
    // output["theta"]
    // output["z"]
    // output["w"]
    // output["gamma"]
    // output["sigma_theta"]
    // output["sigma"]
    // output["map"]
    // output["accept_beta"]
    // output["accept_theta"]
    // output["accept_z"]
    // output["accept_w"]
    // output["accept_gamma"]

    return(output);

} // function end


// You can include R code blocks in C++ files processed with sourceCpp
// (useful for testing and development). The R code will be automatically 
// run after the compilation.




