#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include "util.h"

/* Christopher Prince [cmp670@nyu.edu], 2017
 *
 * A naive numerical solver for solving the 1-D Laplace boundary value problem
 *    -u'' = f,
 *    u(0) = 0, u(1) = 0
 *    f = 1
 *
 * The finite difference approximation for the second derivative is given by
 *    -u''(x_i) ≈ (1/2) * [-u(x_(i-1)) + 2*(u(x_(i))) - u(x_(i+1))]
 * with index i in [0, N-1] for N grid points.
 *
 * This establishes N equations expressed in matrix form as:
 *     (N*N)*Au = f  ==>  Au = f/(N*N)
 * with A the NxN tridiagonal matrix with 2's on the main diagonal, 
 * -1's on the adjacent diagonals, and 0's elsewhere.
 * 
 * Goodness of fit at the k'th iteration is reported as the L2 norm of Au[k] - f.
 *
 * In this implementation, the user can choose between Jacobi and Gauss-Seidel
 * iterative methods using a command line argument ("j" or "g")
 *
 * Usage:
 *
 * $ laplace-iter [N_gridpoints [method [max_iterations [reduction_factor]]]]
 * All options are optional (but must be given in this order), with default arguments
 *     N = 1000
 *     method = "j"
 *     max_iterations = 1000
 *     reduction_factor = 0.0001
 *
 * */

double laplace_L2_norm(double *v, int len);
double *jacobi_iteration(double *u, int len);
double *gs_iteration(double *u, int len);

int main(int argc, char *argv[]){

    int N = 1000;
    char method[] = "j";
    int max_iter = 1000;
    double term_factor = 0.0001;
    char *end;                      //dummy pointer for strtoX()

    // This argument parser is not particularly robust...
    if (argc > 4)
        term_factor = strtod(argv[4], &end);
    if (argc > 3)
        max_iter = atoi(argv[3]);
    if (argc > 2)
        strncpy(method, argv[2], 1);
    if (argc > 1)
        N = atoi(argv[1]);

    if (method[0] != 'j' && method[0] != 'g'){
        fprintf(stderr, "Only valid methods are jacobi and gauss-seidel!\n");
        exit(-1);
    }

    timestamp_type t1, t2;
    get_timestamp(&t1);

    double *u;
    u = malloc(N * sizeof u);

    // initialize first estimate = 0
    for (int i=0; i<N; i++)  u[i] = 0.;
    
    double norm, norm0;

    // calculate L2 norm or residual for initial guess
    norm0 = laplace_L2_norm(u, N);
    norm = norm0;
    fprintf(stderr, "Norm of residual ||Au[0] - f|| = %.8f\n", norm0);

    for (int iter=1; (iter<=max_iter && norm/norm0 > term_factor); iter++){
        switch (method[0]){
            case 'j':
                u = jacobi_iteration(u, N);
                break;
            case 'g':
                u = gs_iteration(u, N);
                break;
        }

        norm = laplace_L2_norm(u, N);
        fprintf(stderr, "Norm of residual ||Au[k] - f|| at iteration %i =  %.8f\n", 
                iter, norm);
    }

    get_timestamp(&t2);

    double elapsed_s = timestamp_diff_in_seconds(t1, t2);
    long elapsed = elapsed_s*1e6;
    fprintf(stderr, "Time elapsed is %li useconds.\n", elapsed);
    fprintf(stderr, "Time elapsed is %f seconds.\n", elapsed_s);
    printf("%li", elapsed);  //is there a better way to do this?

//    for (int i=0; i<N; i++) printf("%.3f\t", u[i]);
//    printf("\n");
    free(u);
    return (1);
}

double *jacobi_iteration(double *u, int len){
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

    double *uu;
    uu = malloc(len * sizeof uu);     // uu is "updated u"

    uu[0] = 0.5*(1/(float)(len*len) + u[1]);
    for (int i=1; i<len-1; i++)
        uu[i] = 0.5*(1/(float)(len*len) + (u[i-1] + u[i+1]));
    uu[len-1] = 0.5*(1/(float)(len*len) + u[len-2]);

    //copy new vector uu into u for next iteration
    for (int i=0; i<len; i++)  u[i]=uu[i];
    
    free(uu);

    return u;
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
     * 
     * This can be updated in-place, so we don't need to create 
     * a separate update vector.
     */

    u[0] = 0.5*(1/(float)(len*len) + u[1]);
    for (int i=1; i<len-1; i++)
        u[i] = 0.5*(1/(float)(len*len) + (u[i-1] + u[i+1]));
    u[len-1] = 0.5*(1/(float)(len*len) + u[len-2]);

    return u;
}

double laplace_L2_norm(double *v, int len){

    //Compute sum of squares of elements in Au - f
    //Note term 1/h^2 (=N^2) in matrix A.
 
    double sumsq = 0;
    sumsq += pow(((2*v[0] - v[1])*(float)(len*len) - 1), 2);
    for (int i=1; i<len-1; i++)
        sumsq += pow(((-v[i-1] + 2*v[i] -v[i+1])*(float)(len*len) - 1), 2);
    sumsq += pow(((-v[len-2] + 2*v[len-1])*(float)(len*len) - 1), 2);

    //The L2 norm is the square root of this sum:
    return sqrt(sumsq);
}
