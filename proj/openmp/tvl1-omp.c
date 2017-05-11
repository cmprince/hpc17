#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "../util.h"
//#include "cl-helper.h"
#include "../ppma_io.h"
#include "tvl1-omp.h"
#include "omp.h"


void nabla(double *img, double *dx, double *dy, int h, int w){

    int i, j, idx;
#pragma omp parallel for private(i) shared(h,w)
    for (i =0; i<h*w; i++){
        dx[i] = 0;
        dy[i] = 0;
    }

#pragma omp parallel for private(i,j) shared(h,w) collapse(2)
    for (i = 0; i < w; i++){
        for (j = 0; j < h; j++){
            idx = j*w + i;
            if (i!=(w-1)){
                dx[idx] -= img[idx];
                dx[idx] += img[idx + 1];
            }
            if (j!=(h-1)){
                dy[idx] -= img[idx];
                dy[idx] += img[idx + w];
            }
        }
    }
}

void nablaT(double *dx, double *dy, double *img, int h, int w){

    int i, j, idx;
#pragma omp parallel for private(i) shared(h,w)
    for (i = 0; i<h*w; i++)
        img[i] = 0;
    
#pragma omp parallel for private (i,j) shared(h,w) collapse(2)
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

void anorm(double *dx, double *dy, double *a, int h, int w){
    
    float sumofsq;
    int i;

//#pragma omp parallel for private(i)
    for (i = 0; i < h*w; i++){
        sumofsq = pow(dx[i], 2) + pow(dy[i], 2);
        a[i] = sqrt(sumofsq);
    }
}

void project(double *dx, double *dy, 
             double *projx, double *projy, 
             double r, double *an, int h, int w){

    
    anorm(dx, dy, an, h, w);
    int i;
#pragma omp parallel for private(i) shared(h,w)
    for (i = 0; i < h*w; i++){
        an[i] = ((an[i]/r > 1.0) ? an[i]/r : 1.0);
        projx[i] = dx[i] / an[i];
        projy[i] = dy[i] / an[i];
    } 
}

double clip(float n, float low, float high){
    return (n<low ? low: (n>high ? high : n));
}

void shrink(double *proj, double *img, double *sh, double step, int h, int w){

    int i;
#pragma omp parallel for private(i) shared(h,w)
    for (i = 0; i < h*w; i++){
            sh[i] = proj[i] + clip(img[i] - proj[i], -step, step);
        }
}

void solve_tvl1(double *img, double *filter, double clambda, int iter, int h, int w){

    double L2 = 8.0;
    double tau = 0.02;
    double theta = 1.0;
    double sigma;
    int i;
    sigma = 1.0 / (float)(L2 * tau);

    double *X, *X1, *Px, *Py, *nablaXx, *nablaXy, *nablaTP, *an;
 
    //X       = calloc(h*w, sizeof(double));
    //X1      = calloc(h*w, sizeof(double));
    //Px      = calloc(h*w, sizeof(double));
    //Py      = calloc(h*w, sizeof(double));
    //nablaXx = calloc(h*w, sizeof(double));
    //nablaXy = calloc(h*w, sizeof(double));
    //nablaTP = calloc(h*w, sizeof(double));
    //an      = calloc(h*w, sizeof(double));
    posix_memalign((void**)&X      , 32, h*w*sizeof(double));
    posix_memalign((void**)&X1     , 32, h*w*sizeof(double));
    posix_memalign((void**)&Px     , 32, h*w*sizeof(double));
    posix_memalign((void**)&Py     , 32, h*w*sizeof(double));
    posix_memalign((void**)&nablaXx, 32, h*w*sizeof(double));
    posix_memalign((void**)&nablaXy, 32, h*w*sizeof(double));
    posix_memalign((void**)&nablaTP, 32, h*w*sizeof(double));
    posix_memalign((void**)&an     , 32, h*w*sizeof(double));

    memcpy(X, img, sizeof *X);

    nabla(X, Px, Py, h, w);
    for (int t = 0; t < iter; t++){
        nabla(X, nablaXx, nablaXy, h, w);

#pragma omp parallel for private(i) shared(h,w,sigma)
        for (i = 0; i < h*w; i++){
            nablaXx[i] *= sigma; 
            nablaXx[i] += Px[i];
            nablaXy[i] *= sigma; 
            nablaXy[i] += Py[i];
        }

        project(nablaXx, nablaXy, Px, Py, 1.0, an, h, w);
        
        nablaT(Px, Py, nablaTP, h, w);

#pragma omp parallel for private(i) shared(h,w,tau)
        for (i = 0; i < h*w; i++){
            nablaTP[i] *= (-1. * tau); 
            nablaTP[i] += X[i];
        }

        shrink(nablaTP, img, X1, clambda*tau, h, w);
        
#pragma omp parallel for private(i) shared(h,w,theta)
        for (i = 0; i < h*w; i++)
            X[i] = X1[i] + theta * (X1[i] - X[i]);
    }

    for (int z=0; z<h*w; z++){
        if (X[z] > 1) {printf("%i: %.2f ", z, X[z]); X[z]=1.;}
        if (X[z] < 0) {printf("%i: %.2f ", z, X[z]); X[z]=0.;}
    }
    
    memcpy(filter, X, h*w* sizeof *X);

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

#pragma omp parallel for private(n) shared(scale,offset)
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

    if(argc != 4)
    {
      fprintf(stderr, "Usage: %s image.ppm num_loops num_threads\n", argv[0]);
      abort();
    }
  
    const char* filename = argv[1];
    const int num_loops = atoi(argv[2]);
    const int num_threads = atoi(argv[3]);
    omp_set_num_threads(num_threads);
    int *r, *g, *b;
    int xsize, ysize, rgb_max;
    int n;
    double *gray, *filter;

    // --------------------------------------------------------------------------
    // load image
    // --------------------------------------------------------------------------
    printf("Reading ``%s''\n", filename);
    ppma_read(filename, &xsize, &ysize, &rgb_max, &r, &g, &b);
    printf("Done reading ``%s'' of size %dx%d\n", filename, xsize, ysize);

    timestamp_type start, finish;
    get_timestamp(&start);

    // --------------------------------------------------------------------------
    // allocate CPU buffers
    // --------------------------------------------------------------------------
    gray   = malloc(xsize * ysize * sizeof (double));
    if(!gray) { fprintf(stderr, "alloc gray"); abort(); }
    filter = malloc(xsize * ysize * sizeof (double));
    if(!filter) { fprintf(stderr, "alloc filter"); abort(); }

    // --------------------------------------------------------------------------
    // convert image to grayscale
    // --------------------------------------------------------------------------
    double rgbmax_inv = 1./rgb_max;
#pragma omp parallel for private(n) shared(rgbmax_inv)
    for(n = 0; n < xsize*ysize; ++n) {
      gray[n] = (0.21*r[n] + 0.72*g[n] + 0.07*b[n])*rgbmax_inv;
    }

    //writeimg(gray, "gray.ppm", ysize, xsize, 1, 0);

    solve_tvl1(gray, filter, 1, num_loops, ysize, xsize);
    writeimg(filter, "output_cpu.ppm", ysize, xsize, 1, 0);

    free(r);
    free(g);
    free(b);
    free(gray);
    free(filter);

    get_timestamp(&finish);
    printf("Elapsed time: %.4f\n", timestamp_diff_in_seconds(start, finish));
}
