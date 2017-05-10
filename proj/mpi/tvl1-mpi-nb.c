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
           uint32_t h, uint32_t w, int rank, uint32_t xprocs, uint32_t yprocs){ 

    int i, j, idx, ghostidx, isRightEdge, isLeftEdge, isDownEdge, isUpEdge;
    double *right, *left, *down, *up;
    right = malloc(h* sizeof right);
    left  = malloc(h* sizeof left);
    up    = malloc(w* sizeof up);
    down  = malloc(w* sizeof up);

    for (int i=0;i<h;++i){right[i]=0;left[i]=0;}
    for (int j=0;j<h;++j){up[j]=0;down[j]=0;}

    MPI_Request req0, req1, req2, req3;

    // is my process responsible for a right or bottom edge?
    isRightEdge = ((rank+1) %            xprocs==0)?1:0;
    isLeftEdge  = ((rank  ) %            xprocs==0)?1:0;
    isDownEdge  = ((rank+1) > (yprocs-1)*xprocs   )?1:0;
    isUpEdge    = ((rank  ) <            xprocs   )?1:0;
printf("got here!");

if (isLeftEdge)
    printf("left\t");
if (isRightEdge)
    printf("right\t");

    //printf("rank %i: r:%i, l:%i, d:%i, u:%i\n", rank, isRightEdge, isLeftEdge, isDownEdge, isUpEdge);
    if (!(isLeftEdge==1)){
        //send my leftmost to my left neighbor
        for (i=0; i<h; ++i)
            left[i] = img[(w+1)*i];
        MPI_Isend(left, h, MPI_DOUBLE, rank-1, 100, MPI_COMM_WORLD, &req0);
    }
    if (!(isRightEdge==1)){
        //receive my right neighbor's leftmost
        MPI_Irecv(right, h, MPI_DOUBLE, rank+1, 100, MPI_COMM_WORLD, &req1);
    }
    if (!(isUpEdge==1)){
        //send my topmost to my up neighbor
        for (i=0; i<w; ++i)
            up[i] = img[i];
        MPI_Isend(up, w, MPI_DOUBLE, rank-xprocs, 100, MPI_COMM_WORLD, &req2);
    }
    if (!(isDownEdge==1)){
        //receive my down neighbor's topmost
        MPI_Irecv(down, w, MPI_DOUBLE, rank+xprocs, 100, MPI_COMM_WORLD, &req3);
    }

    for (i =0; i<(h+1)*(w+1); i++){
        dx[i] = 0;
        dy[i] = 0;
    }
    if (!(isLeftEdge==1))
        MPI_Wait(&req0, MPI_STATUS_IGNORE);
    if (!(isRightEdge==1)){
        MPI_Wait(&req1, MPI_STATUS_IGNORE);
        // place the vector in img's right ghost values
        for (i=0; i<h; ++i)
            img[(w+1)*(i+1)-1] = right[i];
    }
    if (!(isUpEdge==1))
        MPI_Wait(&req2, MPI_STATUS_IGNORE);
    if (!(isDownEdge==1)){
        MPI_Wait(&req3, MPI_STATUS_IGNORE);
        // place the vector in img's bottom ghost values
        for (i=0; i<w; ++i)
            img[(h*(w+1))+i] = down[i];
    }

    for (i = 0; i < w; i++){
        for (j = 0; j < h; j++){
            idx = j*(w+1) + i;
            ghostidx = (j+1)*(w+1)+i+1;
            if (!(isRightEdge && i==w)){
              //  printf("got here");
                dx[ghostidx] -= img[idx];
                dx[ghostidx] += img[idx + 1];
            }
            if (!(isDownEdge && j==h)){
                dy[ghostidx] -= img[idx];
                dy[ghostidx] += img[idx + w + 1];
            }
        }
    }
    free(left);
    free(right);
    free(up);
    free(down);
}

void nablaT(double *dx, double *dy, double *img,
            uint32_t h, uint32_t w, int rank, uint32_t xprocs, uint32_t yprocs){

    int i, j, idx, ghostidx, isRightEdge, isLeftEdge, isDownEdge, isUpEdge;
    double right[h], left[h], down[w], up[w];
    MPI_Request req0, req1, req2, req3;

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
        //receive the righmost dx from my left neighbor
        MPI_Irecv(&left, w, MPI_DOUBLE, rank-1, 102, MPI_COMM_WORLD, &req0);
    }
    if (!isRightEdge){
        //send the rightmost dx to my right neighbor
        for (i=0; i<h; ++i)
            right[i] = dx[(w+1)*(i+1)-1];
        MPI_Isend(&right, w, MPI_DOUBLE, rank+1, 102, MPI_COMM_WORLD, &req1);
    }
    if (!isUpEdge){
        //receive my up neighbor's bottommost dy
        MPI_Irecv(&up, w, MPI_DOUBLE, rank-xprocs, 103, MPI_COMM_WORLD, &req2);
    }
    if (!isDownEdge){
        //send my bottommost dy to my down neighbor
        for (i=0; i<w; ++i)
            down[i] = dy[(w+1)*h+i+1];
        MPI_Isend(&down, w, MPI_DOUBLE, rank+xprocs, 103, MPI_COMM_WORLD, &req3);
    }

    for (i = 0; i<(h+1)*(w+1); i++)
        img[i] = 0;
    
    if (!isLeftEdge)
        MPI_Wait(&req0, MPI_STATUS_IGNORE);
        // place the vector in img's left ghost values
        for (i=0; i<h; ++i)
            dx[(w+1)*(i+1)] = left[i];
    if (!isRightEdge){
        MPI_Wait(&req1, MPI_STATUS_IGNORE);
    }
    if (!isUpEdge)
        MPI_Wait(&req2, MPI_STATUS_IGNORE);
        // place the vector in img's top ghost values
        for (i=0; i<w; ++i)
            dy[i+1] = up[i];
    if (!isDownEdge){
        MPI_Wait(&req3, MPI_STATUS_IGNORE);
    }
    
    for (i = 0; i < w; i++){
        for (j = 0; j < h; j++){
            idx = j * (w+1) + i;
            ghostidx = (j+1)*(w+1)+i+1;
            if (!(isRightEdge && i==0)){
                img[idx] -= dx[ghostidx];
                img[idx] += dx[ghostidx-1];
            }
            if (!(isUpEdge && j==0)){
                img[idx] -= dy[ghostidx];
                img[idx] += dy[ghostidx-w];
            }
        }
    }
}

void project(double *dx, double *dy, 
             double *projx, double *projy, 
             double r, double sigma, double *an, uint32_t h, uint32_t w){

    double sumofsq;
    int i, j, idx, ghostidx;
    for (i = 0; i < w; i++){
        for (j = 0; j < h; j++){
            idx = j*(w+1) + i;
            // Px, Py ghost entries are in the top row and left column!
            ghostidx = (j+1)*(w+1)+i+1;
            dx[ghostidx] *= sigma; 
            dx[ghostidx] += projx[ghostidx];
            dy[ghostidx] *= sigma; 
            dy[ghostidx] += projy[ghostidx];
            sumofsq = pow(dx[ghostidx], 2) + pow(dy[ghostidx], 2);
            an[idx] = sqrt(sumofsq);
            //an[i] = ((an[i]/r > 1.0) ? an[i]/r : 1.0);
            an[idx] = ((an[idx] > 1.0) ? an[idx] : 1.0);
            projx[ghostidx] = dx[ghostidx] / an[idx];
            projy[ghostidx] = dy[ghostidx] / an[idx];
    
        }
    }
}

double clip(float n, float low, float high){
    return (n<low ? low: (n>high ? high : n));
}

void shrink(double *proj, double *img, double *curr, double *sh, 
            double clambda, double tau, double theta, uint32_t h, uint32_t w){

    double step = clambda*tau;
    int i, j, idx, ghostidx;

    for (i = 0; i < w; ++i){
        for (j = 0; j < h; ++j){
            idx = j*(w+1)+i;
            ghostidx = j*w+i;
            proj[idx] *= -(1. * tau);
            proj[idx] += curr[idx];
            sh[idx] = proj[idx] + clip(img[ghostidx] - proj[idx], -step, step);
            // update curr[ent]
            curr[idx] = sh[idx] + theta * (sh[idx] - curr[idx]);
        }
    }
}

void solve_tvl1(double *img, double *filter, double clambda, int iter, 
                uint32_t h, uint32_t w, int rank, uint32_t xprocs, uint32_t yprocs){

    double L2 = 8.0;
    double tau = 0.02;
    double theta = 1.0;
    double sigma;
    sigma = 1.0 / (float)(L2 * tau);
    int idx, ghostidx;

    double *X, *X1, *Px, *Py, *nablaXx, *nablaXy, *nablaTP, *an;
 
    //ghosts on bottom and right:
    posix_memalign((void**)&X      , 32, (h+1)*(w+1)*sizeof(double));
    posix_memalign((void**)&X1     , 32, (h+1)*(w+1)*sizeof(double));
    posix_memalign((void**)&nablaTP, 32, (h+1)*(w+1)*sizeof(double));
    posix_memalign((void**)&an     , 32, (h+1)*(w+1)*sizeof(double));

    //ghosts on top and left:
    posix_memalign((void**)&Px     , 32, (h+1)*(w+1)*sizeof(double));
    posix_memalign((void**)&Py     , 32, (h+1)*(w+1)*sizeof(double));
    posix_memalign((void**)&nablaXx, 32, (h+1)*(w+1)*sizeof(double));
    posix_memalign((void**)&nablaXy, 32, (h+1)*(w+1)*sizeof(double));

    //memcpy(X, img, sizeof *X);
    //Now that I have ghost values in X, I can't just memcpy!
    //TODO: what if we memcpy and just fix when we reassemble parts in main?
    for (int i = 0; i<w; ++i)
        for (int j = 0; j<h; ++j){
            // img is passed in without ghost values, so idx is correct!
            idx = j*w+i;
            ghostidx = j*(w+1)+i;
            X[ghostidx] = img[idx];
        }
            
    char xname[10];
    char ts[5];
    // All the work occurs in this loop:
    nabla(X, Px, Py, h, w, rank, xprocs, yprocs);
    for (int t = 0; t < iter; t++){
        nabla(X, nablaXx, nablaXy, h, w, rank, xprocs, yprocs);
        project(nablaXx, nablaXy, Px, Py, 1.0, sigma, an, h, w);
        nablaT(Px, Py, nablaTP, h, w, rank, xprocs, yprocs);
        shrink(nablaTP, img, X, X1, clambda, tau, theta, h, w);
        strcpy(xname, "X");
        sprintf(ts, "%d", t);
        strcat(xname, ts);
        strcat(xname, ".ppm");
        //if(t%10==0) {writeimg(X, xname, h+1, w+1, 1, 0);}
    }

    for (int z=0; z<(h+1)*(w+1); z++){
        if (X[z] > 1) {X[z]=1.;}
        if (X[z] < 0) {X[z]=0.;}
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


int main(int argc, char *argv[]){

    if(argc != 5)
    {
      fprintf(stderr, "Usage: %s image.ppm num_loops xprocs yprocs\n", argv[0]);
      abort();
    }
  
    const char* filename = argv[1];
    const int num_loops = atoi(argv[2]);
    uint32_t xprocs = (uint32_t)(atoi(argv[3]));
    uint32_t yprocs = (uint32_t)(atoi(argv[4]));
    const int root = 0;
    
    int *r, *g, *b;
    int xsize, ysize, rgb_max, n, p, rank;
    int sub_id, l_idx, g_idx;
    int ii, jj, i, j;
    uint32_t l_sizes[2], xs, ys;
    double *gray, *filter, *mypiece;

	MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &p);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    //MPI_Get_processor_name(hostname, &len);
    //MPI_Request req[2*p];
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

        // convert image to grayscale
        double rgbmax_inv = 1./rgb_max;
        for(n = 0; n < xsize*ysize; ++n) {
          gray[n] = (0.21f*r[n] + 0.72f*g[n] + 0.07f*b[n])*rgbmax_inv;
        }
        
        // calculate subdomain sizes
        int xover = xsize % xprocs;
        int yover = ysize % yprocs;
        uint32_t lx = xsize / xprocs;
        uint32_t ly = ysize / yprocs;
        if (xover!=0)
            printf("dimensions not compatible with processor count; truncating %i columns\n", xover);
        if (yover!=0)
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
        
        xs=lx;
        ys=ly;
        posix_memalign((void**)&mypiece, 32, xs*ys*sizeof(double));
        if(!mypiece) { fprintf(stderr, "alloc mypiece"); abort(); }
        posix_memalign((void**)&filter, 32, xs * ys * sizeof (double));
        if(!filter)  { fprintf(stderr, "alloc filter"); abort(); }
        //memcpy(mypiece, sd[0], sizeof *mypiece);
        for (i=0; i<lx*ly; ++i){mypiece[i] = sd[0][i];}

        uint32_t sizes[2] = {lx, ly};
        // let processors know how much memory to allocate
        for (i = 1; i<p; ++i)
            MPI_Send(&sizes, 2, MPI_UINT32_T, i, 998, MPI_COMM_WORLD); //, &status[i]);
    }

    if(root!=rank){
        // Receive local dimension sizes
        MPI_Recv(&l_sizes, 2, MPI_UINT32_T, 0, 998, MPI_COMM_WORLD, &status[rank]);
        //MPI_Status status;
        //MPI_Probe(0,999,MPI_COMM_WORLD,&status);
        //MPI_Get_count(&status, MPI_DOUBLE, 
        xs = l_sizes[0];
        ys = l_sizes[1];

        posix_memalign((void**)&mypiece, 32, xs*ys*sizeof(double));
        if(!mypiece) { fprintf(stderr, "alloc mypiece"); abort(); }
        posix_memalign((void**)&filter, 32, xs * ys * sizeof (double));
        if(!filter)  { fprintf(stderr, "alloc filter"); abort(); }
        
    }
    
    MPI_Barrier(MPI_COMM_WORLD);

    for (i=1; i<p; ++i){
        if (root==rank){
        // send subdomains to all processors
            MPI_Send(sd[i], xs*ys, MPI_DOUBLE, i, 999, MPI_COMM_WORLD); //, &status[i]);
        }       //end of root==r
        if  (i==rank){
            // receive subdomains
            MPI_Recv(mypiece, xs*ys, MPI_DOUBLE, 0, 999, MPI_COMM_WORLD, &status[rank]);
        }
        MPI_Barrier(MPI_COMM_WORLD);
    }

    timestamp_type start, finish;
    get_timestamp(&start);

    writeimg(mypiece, "mypiece.ppm", ys, xs, 1, 0);

    solve_tvl1(mypiece, filter, 1, num_loops, ys, xs, rank, xprocs, yprocs);
    writeimg(filter, "output_cpu.ppm", ys, xs, 1, 0);

    if (root==rank){
        free(r);
        free(g);
        free(b);
        free(gray);
        for (i=0; i<p; ++i)
            free(sd[i]);
    }
    free(mypiece);
    free(filter);

    MPI_Finalize();

    get_timestamp(&finish);
    printf("Elapsed time: %.4f\n", timestamp_diff_in_seconds(start, finish));
}
