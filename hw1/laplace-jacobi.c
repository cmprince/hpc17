#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>

double laplace_L2_norm(double *v, int len);

int main(int argc, char *argv[]){

    const double term_factor = 0.0001;
    const int max_iter = 1000;
    int N;

    if (argc==1)
        N = 1000;
    else
        N = atoi(argv[1]);

    double h = 1/(double)N;
    printf("%.4f\n", h); 

    // Jacobi iterations:
    // u_i[k+1] = 1/a_ii (f_i - sum(a_ij*u_j[k]; j != i))
    // With
    // a_ij = 2, i=j
    // a_ij = -1, (i-j)^2 = 1
    // a_ij = 0 otherwise
    // 
    // ==> u_i[k+1] = 1/2 (f_i + u_(i-1)[k] + u_(i+1)[k]) 

    //double f[N], u[N], uu[N];       // uu is "updated u"
    
    //allocate on the heap for large N
    double *u, *uu;

    u = malloc(N * sizeof u);
    uu = malloc(N * sizeof uu);     // uu is "updated u"

    for (int i=0; i<N; i++){
        //f[i] = 1.;
        u[i] = 0.;
    }   
    
    double norm0;
    
    // calculate L2 norm or residual for initial guess
    // (i.e., a silly way of calculating sqrt (N))
    norm0 = laplace_L2_norm(u, N);

    for (int iter=0; iter<max_iter; iter++){
        uu[0] = 0.5*(1 + u[1]);
        for (int i=1; i<N-1; i++)
            uu[i] = 0.5*(1 + u[i-1] + u[i+1]);
        uu[N-1] = 0.5*(1 + u[N-2]);

        //for (int i=0; i<N; i++) printf ("%.2f  ", uu[i]);
        
        printf("Norm of residual ||Au[k] - f|| at iteration %i =  %.8f\n", 
                iter, laplace_L2_norm(u, N));

        //copy new vector uu into u for next iteration
        for (int i=0; i<N; i++) u[i]=uu[i];
    }

    free(u);
    free(uu);
}

double laplace_L2_norm(double *v, int len){

    double sumsq = 0;
    sumsq += pow(((2*v[0] - v[1]) - 1), 2);
    for (int i=1; i<len-1; i++)
        sumsq += pow(((-v[i-1] + 2*v[i] -v[i+1]) - 1), 2);
    sumsq += pow(((-v[len-2] + 2*v[len-1]) - 1), 2);

    return sqrt(sumsq);
}
