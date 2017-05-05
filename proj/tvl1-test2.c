#include <stdio.h>
#include <stdlib.h>
#include <math.h>
//#include "timing.h"
//#include "cl-helper.h"
#include "ppma_io.h"
#include "tvl1-test2.h"

#define FILTER_WIDTH 7
#define HALF_FILTER_WIDTH 3

// local size of work group
#define WGX 12
#define WGY 12
#define NON_OPTIMIZED


/* -------------------------------------------------- 
 * Contructor/destructor routines for image structs
 * Adapted from http://stackoverflow.com/questions/14768230/malloc-for-struct-and-pointer-in-c
 * --------------------------------------------------*/

//void freeimg(struct image *img){
//    if (img!=NULL){
//        free(img->data);
//        free(img);
//    }
//}
//
//struct image *makeimg(int height, int width){
//
//    struct image *retVal = malloc (sizeof retVal);
//    if (retVal == NULL)
//        return NULL;
//
//    retVal->data = calloc (height * width, sizeof (double));
//    if (retVal->data == NULL){
//        free (retVal);
//        return NULL;
//    }
//
//    retVal->height = height;
//    retVal->width  = width;
//    return retVal;
//
//}

void nabla(double *img, double *dx, double *dy, int h, int w){
    
    for (int i =0; i<h*w; i++){
        dx[i] = 0;
        dy[i] = 0;
    }

    for (int i = 0; i < w; i++){
        for (int j = 0; j < h; j++){
            int idx = j*w + i;
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
    
    for (int i = 0; i<h*w; i++)
        img[i] = 0;
    
    for (int i = 0; i < w; i++){
        for (int j = 0; j < h; j++){
            int idx = j * w + i;
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

void anorm(double *dx, double *dy, double *a, int h, int w){ //double *a){
    
    float sumofsq;

    for (int i = 0; i < h*w; i++){
        sumofsq = pow(dx[i], 2) + pow(dy[i], 2);
        a[i] = sqrt(sumofsq);
    }
}

void project(double *dx, double *dy, 
             double *projx, double *projy, 
             double r, double *an, int h, int w){

    
    anorm(dx, dy, an, h, w);

    for (int i = 0; i < h*w; i++){
        an[i] = ((an[i]/r < 0) ? an[i]/r : 1.0);
        projx[i] = dx[i] / an[i];
        projy[i] = dy[i] / an[i];
    }
    
}

double clip(float n, float low, float high){
    return (n<low ? low: (n>high ? high : n));
}

void shrink(double *proj, double *img, double *sh, double step, int h, int w){

    for (int j = 0; j < h; j++){
        for (int i = 0; i < w; i++){
            int idx = j*w + i;
            sh[idx] = proj[idx] + clip(img[idx] - proj[idx], -step, step);
        }
    }
}

void solve_tvl1(double *img, double *filter, double clambda, int iter, int h, int w){

    double L2 = 8.0;
    double tau = 0.02;
    double theta = 1.0;
    double sigma;
    sigma = 1.0 / (float)(L2 * tau);

    double *X, *X1, *Px, *Py, *nablaXx, *nablaXy, *nablaTP, *an;
 
    X       = calloc(h*w, sizeof(double));
    X1      = calloc(h*w, sizeof(double));
    Px      = calloc(h*w, sizeof(double));
    Py      = calloc(h*w, sizeof(double));
    nablaXx = calloc(h*w, sizeof(double));
    nablaXy = calloc(h*w, sizeof(double));
    nablaTP = calloc(h*w, sizeof(double));
    an      = calloc(h*w, sizeof(double));

    for (int i=0; i<h*w; i++)
        X[i] = img[i];

    nabla(X, Px, Py, h, w);
    //writeimg(Px, "Px.ppm", h, w, 0.5, 128);
    //writeimg(Py, "Py.ppm", h, w, 0.5, 128);
    for (int t = 0; t < iter; t++){
        nabla(X, nablaXx, nablaXy, h, w);
        for (int i = 0; i < h*w; i++){
            nablaXx[i] *= sigma; 
            nablaXx[i] += Px[i];
            nablaXy[i] *= sigma; 
            nablaXy[i] += Py[i];
        }
        project(nablaXx, nablaXy, Px, Py, 1.0, an, h, w);
        
        nablaT(Px, Py, nablaTP, h, w);
        for (int i = 0; i < h*w; i++){
            nablaTP[i] *= (-1. * tau); 
            nablaTP[i] += X[i];
        }
        shrink(nablaTP, img, X1, clambda*tau, h, w);
        
        for (int i = 0; i < h*w; i++)
            X[i] = X1[i] + theta * (X1[i] - X[i]);
        if (t==iter-1){
            for (int z=0; z<100; z++){printf("%.2f ", X[z]);}
            writeimg(X, "X.ppm", h, w, 1,0);}
    }

    for (int i = 0; i < h*w; i++)
        filter[i] = X[i];

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

    int error;

    const int rgb_max = 255;

    int *red = malloc(h*w*sizeof red);
    int *grn = malloc(h*w*sizeof grn);
    int *blu = malloc(h*w*sizeof blu);

    printf("Writing cpu filtered image\n");
    for(int n = 0; n < h*w; ++n) {
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

// double generateGaussianNoise(const double *mean, const double *stdDev)
// {
// 	static int hasSpare = 0;
//  	static double spare;
//  
//  	if(hasSpare)
//  	{
//  		hasSpare = 0;
//  		return mean + stdDev * spare;
//  	}
//  
//  	hasSpare = 1;
//  	static double u, v, s;
//  	do
//  	{
//  		u = (rand() / ((double) RAND_MAX)) * 2.0 - 1.0;
//  		v = (rand() / ((double) RAND_MAX)) * 2.0 - 1.0;
//  		s = u * u + v * v;
//  	}
//  	while( (s >= 1.0) || (s == 0.0) );
//  
//  	s = sqrt(-2.0 * log(s) / s);
//  	spare = v * s;
//  	return mean + stdDev * u * s;
// }

// void make_noisy(int *img, int N){
//     for (int i = 0; i<N; ++i){
//         img[i] += generateGaussianNoise(0, 0.05); 
//         img[i] = min(1, max(0, img[i]));
// 	}
// }

void main(int argc, char *argv[]){

    if(argc != 3)
    {
      fprintf(stderr, "Usage: %s image.ppm num_loops\n", argv[0]);
      abort();
    }
  
    const char* filename = argv[1];
    const int num_loops = atoi(argv[2]);
    int *r, *g, *b;
    int xsize, ysize, rgb_max;

    double *gray, *filter;

    // --------------------------------------------------------------------------
    // load image
    // --------------------------------------------------------------------------
    printf("Reading ``%s''\n", filename);
    ppma_read(filename, &xsize, &ysize, &rgb_max, &r, &g, &b);
    printf("Done reading ``%s'' of size %dx%d\n", filename, xsize, ysize);
  

    // --------------------------------------------------------------------------
    // allocate CPU buffers
    // --------------------------------------------------------------------------
    //posix_memalign((void**)&gray, 32, xsize*ysize*sizeof(float));
    //posix_memalign((void**)&gray, 32, 4*xsize*ysize*sizeof(float));
    //if(!gray) { fprintf(stderr, "alloc gray"); abort(); }
    //posix_memalign((void**)&tv1, 32, 4*xsize*ysize*sizeof(float));
    //if(!tv1) { fprintf(stderr, "alloc gray"); abort(); }
    //posix_memalign((void**)&congray, 32, 4*xsize*ysize*sizeof(float));
    //if(!congray) { fprintf(stderr, "alloc gray"); abort(); }
    //posix_memalign((void**)&congray_cl, 32, 4*xsize*ysize*sizeof(float));
    //if(!congray_cl) { fprintf(stderr, "alloc gray"); abort(); }
    gray   = malloc(xsize * ysize * sizeof (double));
    //gray = makeimg(xsize, ysize);
    filter = malloc(xsize * ysize * sizeof (double));
    //filter = makeimg(xsize, ysize);

    // --------------------------------------------------------------------------
    // convert image to grayscale
    // --------------------------------------------------------------------------
    for(int n = 0; n < xsize*ysize; ++n) {
      gray[n] = (0.21f*r[n])/rgb_max + (0.72f*g[n])/rgb_max + (0.07f*b[n])/rgb_max;
    }

    writeimg(gray, "gray.ppm", ysize, xsize, 1, 0);

    solve_tvl1(gray, filter, 1, num_loops, ysize, xsize);
    writeimg(filter, "output_cpu.ppm", ysize, xsize, 1, 0);

    free(r);
    free(g);
    free(b);
    free(gray);
    free(filter);

}
