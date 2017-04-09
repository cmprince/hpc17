/***************************
 * 2D Jacobi solver using MPI
 *
 * Christopher Prince
 * cmp670@nyu.edu
 * 13 March 2017
 * ************************/

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "util.h"
#include "mpi.h"

double L2_res(double ***lu, int lN);
void swaparray(double ***u, double ***v);
void jacobi_iteration(double ***u, double ***uu, int lN);

double L2_res(double ***lu, int lN, int x0, int y0){
    //Compute sum of squares of elements in Au - f
    //Note term 1/h^2 (=N^2) in matrix A.
    
    int i, j;
    double tmp, sumsq=0.0, gres2=0.0;

    for (i=x0+1; i<x0+lN+1; i++){
        for (j=y0+1; j<y0+lN+1; j++){
            sumsq += pow(((-(*lu)[i-1][j] -(*lu)[i+1][j] - (*lu)[i][j-1] - (*lu)[i][j+1]
                            + 4*(*lu)[i][j])*(float)(lN*lN) - 1), 2);
        }
    }

    MPI_Allreduce(&sumsq, &gres2, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
    return sqrt(gres2);
}

void swaparray(double ***u, double ***v){
    // Swap 2 2-D arrays of doubles by reference
    double **temp = *v;
    *v = *u;
    *u = temp;
}

void jacobi_iteration(double ***u, double ***uu, int lN){
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
    for (i=1; i<N+1; i++)
        for (j=1; j<M+1; j++)
            (*uu)[i][j] = 0.25*(1/(float)(N*M) + ((*u)[i-1][j] + (*u)[i+1][j] + \
                        (*u)[i][j-1] + (*u)[i][j+1]));
}

int main (int argc, char *argv[]){
    int  r, p       //rank, numprocesses
         i, j,      //iterators
         N, lN,     //side length, submatrix side length
         iter, max_iter;
    int  numtasks, taskid, len, buffer, root, count;
    char hostname[MPI_MAX_PROCESSOR_NAME];

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &p);
    MPI_Comm_rank(MPI_COMM_WORLD, &r);
    MPI_Get_processor_name(hostname, &len);
    root = 0;

    sscanf(argv[1], "%d", &N);
    sscanf(argv[2], "%d", &max_iter);

    //Check to see if the parameters are OK for our simplified assumptions
    //First: is the number of processes a square?
    int s = (int)sqrt(p);   //s = side
    if (s*s <> p){
        printf("Exiting. Number of processes must be a perfect square.\n");
        MPI_Abort(MPI_COMM_WORLD, 0);
    }

    //Second: is the side length divisible by sqrt(p)?
    if ((N % s != 0) && mpirank == 0 ) {
        printf("N: %d, local N: %d\n", N, lN);
        printf("Exiting. N must be a multiple of p\n");
        MPI_Abort(MPI_COMM_WORLD, 0);
    }
    else
        lN = N / s;

    /* timing */
    MPI_Barrier(MPI_COMM_WORLD);
    timestamp_type time1, time2;
    get_timestamp(&time1);

    printf ("Task %d on %s starting...\n", taskid, hostname);
    count = taskid;

    //Scheme (blocking communication):
    //Task 0 determines sizes and cuts
    //Task 0 keeps first part for tself and sends rest to other tasks
    //All tasks compute inner portions of subarrays
    //All tasks send 2, 3 or 4 ghost vectors to neighbors
    //All tasks receive 2, 3 or 4 ghost vectors from neighbors
    //All tasks compute boundaries using ghost vectors
    //All tasks compute its portion of residual
    //All tasks reduce 
    //Repeat!
    //
    //  p0      p1      p2      ...     p[N-1]
    //  pN      pN+1    pN+2    ...     p[2N-1]
    //  p2N     p2N+1   p2N+2   ...     p[3N-1]
    //  ...     ...     ...     ...     ...
    //  pN(N-1) ...     ...     ...     p[N*N-1]
    //
    //  ==>  x0=p/N, y0=p%N

    timestamp_type t1, t2;
    get_timestamp(&t1);

    if (0==r){
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
