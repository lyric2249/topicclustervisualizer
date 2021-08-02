
#include <iostream>
#include <armadillo>

using namespace std;
using namespace arma;


mat A = randn(2,3);
mat B = randn(4,5);

field<mat> F(2,1);
F(0,0) = A;
F(1,0) = B; 

F.print("F:");

F.save("mat_field");
