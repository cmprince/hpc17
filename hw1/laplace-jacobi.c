#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>

double laplace_L2_norm(double *v, int len);
double *jacobi_iteration(double *u, double *uu, int len);
double *gs_iteration(double *u, int len);

int main(int argc, char *argv[]){

    const double term_factor = 0.0001;
    const int max_iter = 1000;
    int N;

    if (argc==1)
        N = 1000;
    else
        N = atoi(argv[1]);

    double h = 1/(double)N;
    double *u, *uu;

    u = malloc(N * sizeof u);
    uu = malloc(N * sizeof uu);     // uu is "updated u"

    // initialize first estimate = 0
    for (int i=0; i<N; i++)  u[i] = 0.;
    
    double norm, norm0;
    
    // calculate L2 norm or residual for initial guess
    // (i.e., a silly way of calculating sqrt (N))
    norm0 = laplace_L2_norm(u, N);
    norm = norm0;
    printf("Norm of residual ||Au[0] - f|| = %.8f\n", norm0);

    for (int iter=1; (iter<=max_iter && norm/norm0 > term_factor); iter++){
        uu = jacobi_iteration(u, uu, N);
        norm = laplace_L2_norm(uu, N);
        printf("Norm of residual ||Au[k] - f|| at iteration %i =  %.8f\n", 
                iter, norm);

        //copy new vector uu into u for next iteration
        for (int i=0; i<N; i++)  u[i]=uu[i];
    }

    free(u);
    free(uu);
}

double *jacobi_iteration(double *u, double *uu, int len){
    /* Jacobi iterations:
     * u_i[k+1] = 1/a_ii (f_i - sum(a_ij*u_j[k]; j != i))
     *
     * With
     * a_ij = 2, i=j
     * a_ij = -1, (i-j)^2 = 1
     * a_ij = 0 otherwise
     * 
     * ==> u_i[k+1] = 1/2 (f_i + u_(i-1)[k] + u_(i+1)[k]) 
     */

    uu[0] = 0.5*(1 + u[1]);
    for (int i=1; i<len-1; i++)
        uu[i] = 0.5*(1 + u[i-1] + u[i+1]);
    uu[len-1] = 0.5*(1 + u[len-2]);

    return uu;
}

double *gs_iteration(double *u, int len){
    /* Gauss-Seidel iterations:
     * u_i[k+1] = 1/a_ii (f_i - (sum(a_ij*u_j[k+1]; j < i) + sum(a_ij*u_j[k]; j > i))
     *
     * With
     * a_ij = 2, i=j
     * a_ij = -1, (i-j)^2 = 1
     * a_ij = 0 otherwise
     * 
     * ==> u_i[k+1] = 1/2 (f_i + uu_(i-1)[k] + u_(i+1)[k])
     * This can be updated in-place, so we don't need to provide 
     * the output vector in the call.
     */

    u[0] = 0.5*(1 + u[1]);
    for (int i=1; i<len-1; i++)
        u[i] = 0.5*(1 + u[i-1] + u[i+1]);
    u[len-1] = 0.5*(1 + u[len-2]);

    return u;
}

double laplace_L2_norm(double *v, int len){

    double sumsq = 0;
    sumsq += pow(((2*v[0] - v[1]) - 1), 2);
    for (int i=1; i<len-1; i++)
        sumsq += pow(((-v[i-1] + 2*v[i] -v[i+1]) - 1), 2);
    sumsq += pow(((-v[len-2] + 2*v[len-1]) - 1), 2);

    return sqrt(sumsq);
}
