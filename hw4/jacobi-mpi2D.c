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
#include <string.h>
#include <unistd.h>
#include <sys/types.h>
#include "util.h"
#include <mpi.h>

double L2_res(double ***lu, int lN, int x0, int y0, int s);
void swaparray(double ***u, double ***v);
void jacobi_iteration(double ***u, double ***uu, int lN, int r, int x0, int y0, int s);

double L2_res(double ***lu, int lN, int x0, int y0, int s){
    //Compute sum of squares of elements in Au - f
    //Note term 1/h^2 (=N^2) in matrix A.
    
    int N = lN * s;
    int i, j;
    double tmp, sumsq=0.0, gres2=0.0;

    for (i=1; i<lN+1; i++){
        for (j=1; j<lN+1; j++){
            sumsq += pow(((-(*lu)[i-1][j] -(*lu)[i+1][j] - (*lu)[i][j-1] - (*lu)[i][j+1]
                            + 4*(*lu)[i][j])*(float)(N*N) - 1), 2);
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

void jacobi_iteration(double ***u, double ***uu, int lN, int r, int x0, int y0, int s){
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

    int N = lN * s;
    int i, j;
    double *vec_in, *vec_out;
    MPI_Status status;
    vec_in = malloc(lN * sizeof(vec_in));
    vec_out = malloc(lN * sizeof(vec_out));

    //Compute interior points
    for (i=2; i<lN; i++)
        for (j=2; j<lN; j++)
            (*uu)[i][j] = 0.25*(1/(float)(N*N) + ((*u)[i-1][j] + (*u)[i+1][j] + \
                        (*u)[i][j-1] + (*u)[i][j+1]));

    //Communicate ghost vectors and update the border
    //Left direction: x0 > 0
    if (x0>0){
        for (i=0; i<lN; i++)
            vec_out[i] = (*u)[i+1][1];
//        MPI_Sendrecv(&vec_out[0], lN, MPI_DOUBLE, y0*s + x0 -1, 100,
//                     &vec_in[0] , lN, MPI_DOUBLE, y0*s + x0 -1, 100,
//                     MPI_COMM_WORLD, status[1]);
        MPI_Send(&vec_out[0], lN, MPI_DOUBLE, y0*s + x0 - 1, 99, MPI_COMM_WORLD);
        MPI_Recv(&vec_in[0],  lN, MPI_DOUBLE, y0*s + x0 - 1, 99, MPI_COMM_WORLD, &status);
        for (i=0; i<lN; i++)
            (*u)[i+1][0] = vec_in[i];
        for (i=1; i<lN+1; i++)
            (*uu)[i][1] = 0.25*(1/(float)(N*N) + ((*u)[i-1][1] + (*u)[i+1][1] + \
                        (*u)[i][0] + (*u)[i][2]));
    }

    //Right direction: x0 < s
    if (x0<(s-1)){
        for (i=0; i<lN; i++)
            vec_out[i] = (*u)[i+1][lN];
//        MPI_Sendrecv(&vec_out[0], lN, MPI_DOUBLE, y0*s + x0 +1, 100,
//                     &vec_in[0] , lN, MPI_DOUBLE, y0*s + x0 +1, 100,
//                     MPI_COMM_WORLD, status[2]);
        MPI_Send(&vec_out[0], lN, MPI_DOUBLE, y0*s + x0 + 1, 99, MPI_COMM_WORLD);
        MPI_Recv(&vec_in[0],  lN, MPI_DOUBLE, y0*s + x0 + 1, 99, MPI_COMM_WORLD, &status);
        for (i=0; i<lN; i++)
            (*u)[i+1][lN+1] = vec_in[i];
        for (i=1; i<lN+1; i++)
            (*uu)[i][lN] = 0.25*(1/(float)(N*N) + ((*u)[i-1][lN] + (*u)[i+1][lN] + \
                        (*u)[i][lN-1] + (*u)[i][lN+1]));
    }

    //Up direction: y0 > 0
    if (y0>0){
        for (i=0; i<lN; i++)
            vec_out[i] = (*u)[1][i+1];
//        MPI_Sendrecv(u[1][1],    lN, MPI_DOUBLE, (y0-1)*s + x0, 100,
//                     u[lN+1][1], lN, MPI_DOUBLE, (y0-1)*s + x0, 100,
//                     MPI_COMM_WORLD, status[3]);
        MPI_Send(&vec_out[0], lN, MPI_DOUBLE, (y0-1)*s + x0, 99, MPI_COMM_WORLD);
        MPI_Recv(&vec_in[0],  lN, MPI_DOUBLE, (y0-1)*s + x0, 99, MPI_COMM_WORLD, &status);
        
        for (i=0; i<lN; i++)
            (*u)[0][i+1] = vec_in[i];
        for (i=1; i<lN+1; i++)
            (*uu)[1][i] = 0.25*(1/(float)(N*N) + ((*u)[1][i-1] + (*u)[1][i+1] + \
                        (*u)[0][i] + (*u)[2][i]));
    }

    //Down direction: y0 < s
    if (y0<(s-1)){
        for (i=0; i<lN; i++)
            vec_out[i] = (*u)[lN][i+1];
//        MPI_Sendrecv(u[lN+1][1],    lN, MPI_DOUBLE, (y0+1)*s + x0, 100,
//                     u[1][1], lN, MPI_DOUBLE, (y0+1)*s + x0, 100,
//                     MPI_COMM_WORLD, status[0]);
        MPI_Send(&vec_out[0],   lN, MPI_DOUBLE, (y0+1)*s + x0, 99, MPI_COMM_WORLD);
        MPI_Recv(&vec_in[0], lN, MPI_DOUBLE, (y0+1)*s + x0, 99, MPI_COMM_WORLD, &status);
        
        for (i=0; i<lN; i++)
            (*u)[lN+1][i+1] = vec_in[i];
        for (i=1; i<lN+1; i++)
            (*uu)[lN][i] = 0.25*(1/(float)(N*N) + ((*u)[lN][i-1] + (*u)[lN][i+1] + \
                        (*u)[lN-1][i] + (*u)[lN+1][i]));
    }

    free (vec_in);
    free (vec_out);
}

int main (int argc, char *argv[]){
    int    r, p,       //rank, numprocesses
           i, j,      //iterators
           N, lN,     //side length, submatrix side length
           iter, max_iter;
    int    numtasks, taskid, len, buffer, root, count;
    double l2, term_factor=0.1;
    char   hostname[MPI_MAX_PROCESSOR_NAME];

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &p);
    MPI_Comm_rank(MPI_COMM_WORLD, &r);
    MPI_Get_processor_name(hostname, &len);
    root = 0;

    //debug loop, attach with gdb --pid <number>
    if(0==r)
    {
        int i = 0;
        char hostname[256];
        gethostname(hostname, sizeof(hostname));
        printf("PID %d on %s ready for attach\n", getpid(), hostname);
        fflush(stdout);
        while (0 == i)
            sleep(5);
    }

    sscanf(argv[1], "%i", &N);
    sscanf(argv[2], "%i", &max_iter);
    
//    N = atoi(argv[1]);
//    max_iter = atoi(argv[2]);
    
    //Check to see if the parameters are OK for our simplified assumptions
    //First: is the number of processes a square?
    int s = (int)sqrt(p);   //s = side
    if (s*s != p){
        printf("Exiting. Number of processes must be a perfect square.\n");
        MPI_Abort(MPI_COMM_WORLD, 0);
    }

    //Second: is the side length divisible by sqrt(p)?
    if ((N % s != 0) && r == 0 ) {
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

    printf ("Task %i on %s starting...\n", r, hostname);
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
    //  p0      p1      p2      ...     p[s-1]
    //  ps      ps+1    ps+2    ...     p[2s-1]
    //  p2s     p2s+1   p2s+2   ...     p[3s-1]
    //  ...     ...     ...     ...     ...
    //  ps(s-1) ...     ...     ...     p[s*s-1]
    //
    //  ==>  x0 = r%s, y0 = r/s

    timestamp_type t1, t2;
    get_timestamp(&t1);

    double **u;
    // +2 on lN to add a boundary
    u =  malloc((lN+2) * sizeof *u);
    if (u)
        for (int i=0; i<N+2; i++)
            u[i] = malloc((lN+2)* sizeof *u[i]);
    // initialize first estimate = 0
    for (int i=0; i<lN+2; i++)  
        for (int j=0; j<lN+2; j++)
            u[i][j] = 0.;
    double norm, norm0;

    int x0 = r%s;
    int y0 = r/s;

    // calculate L2 norm or residual for initial guess
    l2 = L2_res(&u, lN, x0, y0, s);
    norm0 = l2;
    if (0==r) 
        fprintf(stderr, "Norm of residual ||Au[0] - f|| = %.8f\n", norm0);
    norm = norm0;

    // +2 on lN to add a boundary
    double **uu;
    uu = malloc((lN+2) * sizeof *uu);
    if (uu)
        for (int i=0; i<(lN+2); i++)
            uu[i] = malloc((lN+2)* sizeof *uu[i]);

    for (int i=0; i<lN+2; i++)  
        for (int j=0; j<lN+2; j++)
            uu[i][j] = 0.;

    for (int iter=1; (iter<=max_iter && norm/norm0 > term_factor); iter++){
        MPI_Barrier(MPI_COMM_WORLD);
        jacobi_iteration(&u, &uu, lN, r, x0, y0, s);
        swaparray(&u, &uu);
        l2 = L2_res(&u, lN, x0, y0, s);
        if (!(iter%10)&& 0==r)
            fprintf(stderr, "Norm of residual ||Au[k] - f|| at iteration %i =  %.8f\n", iter, l2);
    }

/*    for (int i=0; i<(N+2); i++)
        free (uu[i]);
    free(uu);
    for (int i=0; i<(N+2); i++)
        free (u[i]);
    free(u);
*/
    if (0==r){
        get_timestamp(&t2);

    double elapsed_s = timestamp_diff_in_seconds(t1, t2);
    long elapsed = elapsed_s*1e6;
    fprintf(stderr, "Time elapsed is %li useconds.\n", elapsed);
    fprintf(stderr, "Time elapsed is %f seconds.\n", elapsed_s);
    printf("%li", elapsed);  //is there a better way to do this?
    }
MPI_Finalize();
    }

