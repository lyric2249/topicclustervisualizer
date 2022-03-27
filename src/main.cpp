#include <cstdlib>
#include <string.h>
#include <string>
#include <cstdlib>
#include <string.h>
#include <string>
#include <iostream>
#include <ctime>

#include "model.h"
#include "infer.h"

using namespace std;



//// checksum
/*
  if (argc < 4) {
     usage();
     return 1;
   }
*/



//// estimate mode
//   if (strcmp(argv[i++], "est")==0) {
int main()
{
    //// load parameters from std input
  int i = 1;

  int K = atoi(argv[i++]);                  // topic num
  int W = atoi(argv[i++]);
  double alpha = atof(argv[i++]);    // hyperparameters of p(z)
  double beta = atof(argv[i++]);     // hyperparameters of p(w|z)
  int n_iter = atoi(argv[i++]);
  int save_step = atoi(argv[i++]);
  
  string docs_pt(argv[i++]);
  string dir(argv[i++]);

  cout << "Run BTM, K=" << K << ", W=" << W << ", alpha=" << alpha << ", beta=" << beta << ", n_iter=" << n_iter << ", save_step=" << save_step << " ====" << endl;	
  // load training data from file
  clock_t start = clock();
  Model model(K, W, alpha, beta, n_iter, save_step);
  model.run(docs_pt, dir);
  clock_t end = clock();
  printf("cost %fs\n", double(end - start)/CLOCKS_PER_SEC);	

}





//// inference mode
//  else if (strcmp(argv[1], "inf")==0)
int main(int argc, char* argv[]) 
{
  int i = 1;
  
  string type(argv[2]);
  int K = atoi(argv[3]);                  // topic num
  string docs_pt(argv[4]);
  string dir(argv[5]);
  cout << "Run inference:K=" << K << ", type " << type << " ====" << endl;
  Infer inf(type, K);
  inf.run(docs_pt, dir);

}
