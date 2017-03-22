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

double laplace_L2_norm(double **v, int N, int M);
double **jacobi_iteration(double **u, double **uu, int N, int M);
double **gs_iteration(double **u, int N, int M);
void swaparray(double ***u, double ***v);

int main(int argc, char *argv[]){

    int N = 100;
    int M = 100;
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
        N = M = atoi(argv[1]);

    if (method[0] != 'j' && method[0] != 'g'){
        fprintf(stderr, "Only valid methods are jacobi and gauss-seidel!\n");
        exit(-1);
    }

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
    norm0 = laplace_L2_norm(u, N, M);
    norm = norm0;
    fprintf(stderr, "Norm of residual ||Au[0] - f|| = %.8f\n", norm0);

    switch (method[0]){
        case 'j':
            {
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
                    u = jacobi_iteration(u, uu, N, M);
                    norm = laplace_L2_norm(u, N, M);
                    fprintf(stderr, "Norm of residual ||Au[k] - f|| at iteration %i =  %.8f\n", iter, norm);
                }

                for (int i=0; i<(N+2); i++)
                    free (uu[i]);
                free(uu);

                break;
            }
        case 'g':
            for (int iter=1; (iter<=max_iter && norm/norm0 > term_factor); iter++){
                u = gs_iteration(u, N, M);
                norm = laplace_L2_norm(u, N, M);
                fprintf(stderr, "Norm of residual ||Au[k] - f|| at iteration %i =  %.8f\n", 
                        iter, norm);
            }
            break;

    }

    get_timestamp(&t2);

    double elapsed_s = timestamp_diff_in_seconds(t1, t2);
    long elapsed = elapsed_s*1e6;
    fprintf(stderr, "Time elapsed is %li useconds.\n", elapsed);
    fprintf(stderr, "Time elapsed is %f seconds.\n", elapsed_s);
    printf("%li", elapsed);  //is there a better way to do this?

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

double **jacobi_iteration(double **u, double **uu, int N, int M){
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

    for (int i=1; i<N+1; i++)
        for (int j=1; j<M+1; j++)
            uu[i][j] = 0.25*(1/(float)(N*M) + (u[i-1][j] + u[i+1][j] + 
                        u[i][j-1] + u[i][j+1]));

    return uu;
}

double **gs_iteration(double **u, int N, int M){
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
    int i, j;
    //red-black seperation:
    //black:
#pragma omp parallel for shared(u, N, M) private (i, j)
    for (i=1; i<N+1; i++)
        for (j=i%2+1; j<M+1; j+=2)
            u[i][j] = 0.25*(1/(float)(N*M) + (u[i-1][j] + u[i+1][j] +
                        u[i][j-1] + u[i][j+1]));
    //red:
#pragma omp parallel for shared(u, N, M) private (i, j)
    for (i=2; i<N+1; i+=2)
        for (j=i%2+1; j<M+1; j+=2)
            u[i][j] = 0.25*(1/(float)(N*M) + (u[i-1][j] + u[i+1][j] +
                        u[i][j-1] + u[i][j+1]));
    
    return u;
}

double laplace_L2_norm(double **v, int N, int M){

    //Compute sum of squares of elements in Au - f
    //Note term 1/h^2 (=N^2) in matrix A.
 
    double sumsq = 0;
    for (int i=1; i<N+1; i++)
        for (int j=1; j<M+1; j++)
            sumsq += pow(((-v[i-1][j] - v[i+1][j] - v[i][j-1] - v[i][j+1]
                            + 4*v[i][j])*(float)(N*M) - 1), 2);
    //The L2 norm is the square root of this sum:
    //
    return sqrt(sumsq);
}
