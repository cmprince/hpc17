#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../util.h"
//#include "cl-helper.h"
#include "../ppma_io.h"
#include "tvl1-mpi.h"
#include <inttypes.h>
#include <mpi.h>

void nabla(double *img, double *dx, double *dy,
           int h, int w, int rank, int xprocs, int yprocs){ 

    int i, j, idx, isRightEdge, isLeftEdge, isDownEdge, isUpEdge;
    double right[h], left[h], down[w], up[w];
    MPI_Request req[2];

    // is my process responsible for a right or bottom edge?
    isRightEdge = ((rank+1) %            xprocs==0)?1:0;
    isLeftEdge  = ((rank  ) %            xprocs==0)?1:0;
    isDownEdge  = ((rank+1) > (yprocs-1)*xprocs   )?1:0;
    isUpEdge    = ((rank  ) <            xprocs   )?1:0;

    if (!isLeftEdge){
        //send my leftmost to my left neighbor
        for (i=0; i<h; ++i)
            left[i] = img[w*i];
        MPI_Isend(&left, h, MPI_DOUBLE, rank-1, 100, MPI_COMM_WORLD, &req[0]);
    }
    if (!isRightEdge){
        //receive my right neighbor's leftmost
        MPI_Irecv(&right, h, MPI_DOUBLE, rank+1, 100, MPI_COMM_WORLD, &req[0]);
        // place the vector in img's right ghost values
        for (i=0; i<h; ++i)
            img[(w+1)*(i+1)-1] = right[i];
    }
    if (!isUpEdge){
        //send my topmost to my up neighbor
        for (i=0; i<w; ++i)
            up[i] = img[i];
        MPI_Isend(&up, w, MPI_DOUBLE, rank-xprocs, 100, MPI_COMM_WORLD, &req[1]);
    }
    if (!isDownEdge){
        //receive my down neighbor's topmost
        MPI_Irecv(&down, w, MPI_DOUBLE, rank+xprocs, 100, MPI_COMM_WORLD, &req[1]);
        // place the vector in img's bottom ghost values
        for (i=0; i<w; ++i)
            img[(h*(w+1))+i] = down[i];
    }

    for (i =0; i<h*w; i++){
        dx[i] = 0;
        dy[i] = 0;
    }

    for (i = 0; i < w; i++){
        for (j = 0; j < h; j++){
            int idx = j*(w+1) + i;
            if (!(isRightEdge && i!=(w-1))){
                dx[idx] -= img[idx];
                dx[idx] += img[idx + 1];
            }
            if (!(isDownEdge && j!=(h-1))){
                dy[idx] -= img[idx];
                dy[idx] += img[idx + w];
            }
        }
    }
}

void nablaT(double *dx, double *dy, double *img,
            int h, int w, int rank, int xprocs, int yprocs){

    int i, j, idx, isRightEdge, isLeftEdge, isDownEdge, isUpEdge;
    double right[h], left[h], down[w], up[w];
    MPI_Request req[2];

    // is my process responsible for a right or bottom edge?
    isRightEdge = ((rank+1) %            xprocs==0)?1:0;
    isLeftEdge  = ((rank  ) %            xprocs==0)?1:0;
    isDownEdge  = ((rank+1) > (yprocs-1)*xprocs   )?1:0;
    isUpEdge    = ((rank  ) <            xprocs   )?1:0;

//TODO: send the Px, Py around
//TODO: fix ghostidx here (use project's)
//TODO: Rework update logic for local work only
//TODO: update shrink similarly

    if (!isLeftEdge){
        //send my leftmost to my left neighbor
        for (i=0; i<h; ++i)
            left[i] = img[w*i];
        MPI_Isend(&left, h, MPI_DOUBLE, rank-1, 100, MPI_COMM_WORLD, &req[0]);
    }
    if (!isRightEdge){
        //receive my right neighbor's leftmost
        MPI_Irecv(&right, h, MPI_DOUBLE, rank+1, 100, MPI_COMM_WORLD, &req[0]);
        // place the vector in img's right ghost values
        for (i=0; i<h; ++i)
            img[(w+1)*(i+1)-1] = right[i];
    }
    if (!isUpEdge){
        //send my upmost to my up neighbor
        for (i=0; i<w; ++i)
            up[i] = img[i];
        MPI_Isend(&up, w, MPI_DOUBLE, rank-xprocs, 100, MPI_COMM_WORLD, &req[1]);
    }
    if (!isDownEdge){
        //receive my down neighbor's topmost
        MPI_Irecv(&down, w, MPI_DOUBLE, rank+xprocs, 100, MPI_COMM_WORLD, &req[1]);
        // place the vector in img's bottom ghost values
        for (i=0; i<w; ++i)
            img[(h*(w+1))+i] = down[i];
    }

    for (i = 0; i<h*w; i++)
        img[i] = 0;
    
    for (i = 0; i < w; i++){
        for (j = 0; j < h; j++){
            idx = j * w + i;
            if (i!=(w-1)){
                img[idx]     -= dx[idx];
                img[idx + 1] += dx[idx];
            }
            if (j!=(h-1)){
                img[idx]     -= dy[idx];
                img[idx + w] += dy[idx];
            }
        }
    }
}

void project(double *dx, double *dy, 
             double *projx, double *projy, 
             double r, double sigma, double *an, int h, int w){

    double sumofsq;
    int i, j, idx, ghostidx;
    for (i = 0; i < h; i++){
        for (j = 0; j < w; i++){
            idx = j*w + i;
            // Px, Py ghost indices are in the top row and left column!
            ghostidx = (j+1)*(w+1)+i+1;
            dx[idx] *= sigma; 
            dx[idx] += projx[ghostidx];
            dy[idx] *= sigma; 
            dy[idx] += projy[ghostidx];
            sumofsq = pow(dx[idx], 2) + pow(dy[idx], 2);
            an[idx] = sqrt(sumofsq);
            //an[i] = ((an[i]/r > 1.0) ? an[i]/r : 1.0);
            an[idx] = ((an[idx] > 1.0) ? an[idx] : 1.0);
            projx[ghostidx] = dx[idx] / an[idx];
            projy[ghostidx] = dy[idx] / an[idx];
    }
    
}

double clip(float n, float low, float high){
    return (n<low ? low: (n>high ? high : n));
}

void shrink(double *proj, double *img, double *curr, double *sh, 
            double clambda, double tau, double theta, int h, int w){

    double step = clambda*tau;
    int i;
    for (i = 0; i < h * w; i++){
        proj[i] *= -(1. * tau);
        proj[i] += curr[i];
        sh[i] = proj[i] + clip(img[i] - proj[i], -step, step);
        // update curr[ent]
        curr[i] = sh[i] + theta * (sh[i] - curr[i]);
    }
}

void solve_tvl1(double *img, double *filter, double clambda, int iter, 
                int h, int w, int rank, int xprocs, int yprocs){

    double L2 = 8.0;
    double tau = 0.02;
    double theta = 1.0;
    double sigma;
    sigma = 1.0 / (float)(L2 * tau);
    int idx, ghostidx;

    double *X, *X1, *Px, *Py, *nablaXx, *nablaXy, *nablaTP, *an;
 
    posix_memalign((void**)&X      , 32, (h+1)*(w+1)*sizeof(double));
    posix_memalign((void**)&X1     , 32, h*w*sizeof(double));
    posix_memalign((void**)&Px     , 32, (h+1)*(w+1)*sizeof(double));
    posix_memalign((void**)&Py     , 32, (h+1)*(w+1)*sizeof(double));
    posix_memalign((void**)&nablaXx, 32, h*w*sizeof(double));
    posix_memalign((void**)&nablaXy, 32, h*w*sizeof(double));
    posix_memalign((void**)&nablaTP, 32, h*w*sizeof(double));
    posix_memalign((void**)&an     , 32, h*w*sizeof(double));

    //memcpy(X, img, sizeof *X);
    //Now that I have ghost values in X, I can't just memcpy!
    //TODO: what if we memcpy and just fix when we reassemble parts in main?
    for (int i = 0; i<w; ++i)
        for (int j = 0; j<h; ++j){
            idx = j*w+i;
            ghostidx = j*(w+1)+i;
            X[ghostidx] = img[idx];
        }
            

    // All the work occurs in this loop:
    nabla(X, Px, Py, h, w, rank, xprocs, yprocs);
    for (int t = 0; t < iter; t++){
        nabla(X, nablaXx, nablaXy, h, w, rank, xprocs, yprocs);
        project(nablaXx, nablaXy, Px, Py, 1.0, sigma, an, h, w);
        nablaT(Px, Py, nablaTP, h, w, rank, xprocs, yprocs);
        shrink(nablaTP, img, X, X1, clambda, tau, theta, h, w);
    }

    for (int z=0; z<h*w; z++){
        if (X[z] > 1) {printf("%i: %.2f ", z, X[z]); X[z]=1.;}
        if (X[z] < 0) {printf("%i: %.2f ", z, X[z]); X[z]=0.;}
    }

    //memcpy(filter, X, h*w* sizeof *X);
    //Ditto above! Need to account for ghost values
    for (int i = 0; i<w; ++i)
        for (int j = 0; j<h; ++j){
            idx = j*w+i;
            ghostidx = j*(w+1)+i;
            filter[idx] = X[ghostidx];
        }

    free(X);
    free(X1);
    free(Px);
    free(Py);
    free(nablaXx);
    free(nablaXy);
    free(nablaTP);
    free(an);
}

void writeimg(double *img, char *fname, int h, int w, double scale, int offset){
    // --------------------------------------------------------------------------
    // output cpu filtered image
    // --------------------------------------------------------------------------

    int error, n;

    const int rgb_max = 255;

    int *red = malloc(h*w*sizeof red);
    int *grn = malloc(h*w*sizeof grn);
    int *blu = malloc(h*w*sizeof blu);

    printf("Writing cpu filtered image\n");
    for(n = 0; n < h*w; ++n) {
      red[n] = (int)(img[n] * rgb_max * scale + offset);
      grn[n] = (int)(img[n] * rgb_max * scale + offset);
      blu[n] = (int)(img[n] * rgb_max * scale + offset);
    }
    error = ppma_write(fname, w, h, red, grn, blu);
    if(error) { fprintf(stderr, "error writing image"); abort(); }

    free(red);
    free(grn);
    free(blu);

}


void main(int argc, char *argv[]){

    if(argc != 5)
    {
      fprintf(stderr, "Usage: %s image.ppm num_loops xprocs yprocs\n", argv[0]);
      abort();
    }
  
    const char* filename = argv[1];
    const int num_loops = atoi(argv[2]);
    int xprocs = atoi(argv[3]);
    int yprocs = atoi(argv[4]);
    const int root = 0;

    int *r, *g, *b;
    int xsize, ysize, rgb_max, n, p, rank;
    int sub_id, l_idx, g_idx;
    int ii, jj, i, j;
    int isRightEdge, isDownEdge;
    uint16_t l_sizes[2], xs, ys;
    double *gray, *filter, *mypiece;

	MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &p);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    //MPI_Get_processor_name(hostname, &len);
    MPI_Request req[p];
    MPI_Status status[p];
    double *sd[p];

 //debug loop, attach with gdb --pid <number>
    if(999999==rank)
    {
        i = 0;
        char hostname[256];
        gethostname(hostname, sizeof(hostname));
        printf("PID %d on %s ready for attach\n", getpid(), hostname);
        fflush(stdout);
        while (0 == i)
            sleep(5);
    }
    // --------------------------------------------------------------------------
    // preliminaries for the root process to take care of
    // --------------------------------------------------------------------------
 	if (root==rank){
        // abort if proc layout is not properly specified
        if (xprocs*yprocs != p){
            printf("Number of processors is not equal to the product of xprocs and yprocs!\n");
            MPI_Finalize();
            abort();
        }
        
        // load image
        printf("Reading ``%s''\n", filename);
        ppma_read(filename, &xsize, &ysize, &rgb_max, &r, &g, &b);
        printf("Done reading ``%s'' of size %dx%d\n", filename, xsize, ysize);
        

        // allocate CPU buffers
        posix_memalign((void**)&gray, 32, xsize*ysize*sizeof(double));
        if(!gray) { fprintf(stderr, "alloc gray"); abort(); }
        posix_memalign((void**)&filter, 32, xsize * ysize * sizeof (double));
        if(!filter) { fprintf(stderr, "alloc filter"); abort(); }

        // convert image to grayscale
        double rgbmax_inv = 1./rgb_max;
        for(n = 0; n < xsize*ysize; ++n) {
          gray[n] = (0.21f*r[n] + 0.72f*g[n] + 0.07f*b[n])*rgbmax_inv;
        }

        // calculate subdomain sizes
        int xover = xsize % xprocs;
        int yover = ysize % yprocs;
        uint16_t lx = xsize / xprocs;
        uint16_t ly = ysize / yprocs;
        if (xover)
            printf("dimensions not compatible with processor count; truncating %i columns\n", xover);
        if (yover)
            printf("dimensions not compatible with processor count; truncating %i rows\n", yover);
        
        // allocate subdomain buffers
        for (int i = 0; i<p; ++i)
            posix_memalign((void**)&sd[i], 32, lx*ly*sizeof(double));

        // create subdomains
        for (ii = 0; ii<xprocs; ++ii){
            for (jj = 0; jj<yprocs; ++jj){
                sub_id = jj*xprocs + ii;
                for (i = 0; i<lx; ++i){
                    for (j = 0; j<ly; ++j){
                        l_idx = j*lx + i;
                        g_idx = (jj*ly+j)*xsize + ii*lx + i;
                        sd[sub_id][l_idx] = gray[g_idx];
                    }
                }
            }
        }
        
        uint16_t sizes[2] = {lx, ly};
        // let processors know how much memory to allocate
        for (i = 0; i<p; ++i)
            MPI_Isend(&sizes, 2, MPI_UINT16_T, i, 998, MPI_COMM_WORLD, &req);
        // send subdomains to all processors
        for (i = 0; i<p; ++i)
            MPI_Isend(sd[i], (int)(lx*ly), MPI_DOUBLE, i, 999, MPI_COMM_WORLD, &req);
    }       //end of root==r

    // Receive local dimension sizes
    MPI_Irecv(&l_sizes, 2, MPI_UINT16_T, 0, 998, MPI_COMM_WORLD, &req);
    xs = l_sizes[0];
    ys = l_sizes[1];

    posix_memalign((void**)&mypiece, 32, xs*ys*sizeof(double));
    
    // receive subdomains
    MPI_Irecv(mypiece, (int)(xs*ys), MPI_DOUBLE, 0, 999, MPI_COMM_WORLD, &req);

    MPI_Barrier(MPI_COMM_WORLD);

    timestamp_type start, finish;
    get_timestamp(&start);

    //writeimg(gray, "gray.ppm", ysize, xsize, 1, 0);

    solve_tvl1(mypiece, filter, 1, num_loops, ys, xs, rank, xprocs, yprocs);
    writeimg(filter, "output_cpu.ppm", ys, xs, 1, 0);

    if (root==rank){
        free(r);
        free(g);
        free(b);
        free(gray);
        free(filter);
        for (i=0; i<p; ++i)
            free(sd[i]);
    }
    
    MPI_Finalize();

    get_timestamp(&finish);
    printf("Elapsed time: %.4f\n", timestamp_diff_in_seconds(start, finish));
}
