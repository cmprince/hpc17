#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <string.h>
#include <omp.h>
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

void laplace_L2_norm(double ***v, int N, int M, double *l2);
void jacobi_iteration(double ***u, double ***uu, int N, int M);
void swaparray(double ***u, double ***v);

double l2, sumsq;

int main(int argc, char *argv[]){

    int N = 100;
    int M = 100;
    int max_iter = 1000;
    double term_factor = 0.0001;
    char *end;                      //dummy pointer for strtoX()

    // This argument parser is not particularly robust...
    if (argc > 3)
        term_factor = strtod(argv[3], &end);
    if (argc > 2)
        max_iter = atoi(argv[2]);
    if (argc > 1)
        N = M = atoi(argv[1]);

    timestamp_type t1, t2;
    get_timestamp(&t1);

    double **u;
    // +2 on N and M to add a boundary
    u =  malloc((N+2) * sizeof *u);

    if (u)
        for (int i=0; i<N+2; i++)
            u[i] = malloc((M+2)* sizeof *u[i]);

    // initialize first estimate = 0
    for (int i=0; i<N+2; i++)  
        for (int j=0; j<M+2; j++)
            u[i][j] = 0.;
    
    double norm, norm0;

    // calculate L2 norm or residual for initial guess
    laplace_L2_norm(&u, N, M, &l2);
    norm0 = l2;
    fprintf(stderr, "Norm of residual ||Au[0] - f|| = %.8f\n", norm0);
    norm = norm0;

    // +2 on N and M to add a boundary
    double **uu;
    uu = malloc((N+2) * sizeof *uu);
    if (uu)
        for (int i=0; i<(N+2); i++)
            uu[i] = malloc((M+2)* sizeof *uu[i]);

    for (int i=0; i<N+2; i++)  
        for (int j=0; j<M+2; j++)
            uu[i][j] = 0.;

    for (int iter=1; (iter<=max_iter && norm/norm0 > term_factor); iter++){
        jacobi_iteration(&u, &uu, N, M);
        swaparray(&u, &uu);
        laplace_L2_norm(&u, N, M, &l2);
        if (!(iter%100))
            fprintf(stderr, "Norm of residual ||Au[k] - f|| at iteration %i =  %.8f\n", iter, l2);
    }

    for (int i=0; i<(N+2); i++)
        free (uu[i]);
    free(uu);

    get_timestamp(&t2);

    double elapsed_s = timestamp_diff_in_seconds(t1, t2);
    long elapsed = elapsed_s*1e6;
    fprintf(stderr, "Time elapsed is %li useconds.\n", elapsed);
    fprintf(stderr, "Time elapsed is %f seconds.\n", elapsed_s);
    printf("%li", elapsed);  //is there a better way to do this?

// I seem to have problems freeing the first array, probably because I'm swapping the
// pointers with reckless abandon. We're at the end of the program, so we're pretty
// safe ignoring it and letting the OS reclaim the memory at exit.
//
//    for (int i=0; i<N+2; i++)
//        free (u[i]);
//    free (u);
    return (1);
}

void swaparray(double ***u, double ***v){
    // Swap 2 2-D arrays of doubles by reference
    double **temp = *v;
    *v = *u;
    *u = temp;
}

void jacobi_iteration(double ***u, double ***uu, int N, int M){
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
    int i, j;
#pragma omp parallel for shared(u, uu, M, N) private (i, j)
    for (i=1; i<N+1; i++)
        for (j=1; j<M+1; j++)
            (*uu)[i][j] = 0.25*(1/(float)(N*M) + ((*u)[i-1][j] + (*u)[i+1][j] + \
                        (*u)[i][j-1] + (*u)[i][j+1]));
}

void laplace_L2_norm(double ***v, int N, int M, double *l2){

    //Compute sum of squares of elements in Au - f
    //Note term 1/h^2 (=N^2) in matrix A.
 
    int i, j;
    double d;
    sumsq=0.;

#pragma omp parallel for reduction(+: sumsq) private(i, j, d) shared(N, M, v)
    for (i=1; i<N+1; i++)
        for (j=1; j<M+1; j++){
            d = ((-(*v)[i-1][j] - (*v)[i+1][j] - (*v)[i][j-1] - (*v)[i][j+1] + 4*(*v)[i][j])*(float)(N*M) - 1);
            sumsq += d*d;
        }
    //The L2 norm is the square root of this sum:
    //
    *l2 = sqrt(sumsq);
}
