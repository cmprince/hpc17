#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>

int main(int argc, char *argv[]){

    const double term_factor = 0.0001;
    const int max_iter = 1000;
    int N;

    if (argc==1)
        N = 1000;
    else
        N = atoi(argv[1]);

    double h = 1/(double)N;
    
    // Jacobi iterations:
    // u_i[k+1] = 1/a_ii (f_i - sum(a_ij*u_j[k]; j != i))
    // With
    // a_ij = 2, i=j
    // a_ij = -1, (i-j)^2 = 1
    // a_ij = 0 otherwise
    // 
    // ==> u_i[k+1] = 1/2 (f_i - u_(i-1)[k] - u_(i+1)[k]) 

    double f[N], u[N], uu[N];
    for (int i=0; i<N; i++){
        f[i] = 1.;
        u[i] = 0.;
    }

    for (int iter=0; iter<max_iter; iter++){
        uu[0] = 1/2.*(f[0] - u[1]);
        for (int i=1; i<N-1; i++){
            uu[i] = 1/2.*(f[i] - u[i-1] - u[i+1]);
        }
        uu[N] = 1/2.*(f[N] - u[N-1]);

        //calculate norm
        double sumsq = 0;
        sumsq += pow(((2*u[0] - u[1])/pow(h,2) - f[0]), 2);
        for (int i=1; i<N-1; i++)
            sumsq += pow(((-u[i-1] + 2*u[i] -u[i+1])/pow(h,2) - f[1]), 2);
        sumsq += pow(((-u[N-1] + 2*u[N]) -f[N])/pow(h,2), 2);
        
        printf("Norm of residual ||Au[k] - f|| = %.8f\n", sqrt(sumsq));

        //copy new vector uu into u for next iteration
        for (int i=0; i<N; i++) u[i]=uu[i];
    }

}

