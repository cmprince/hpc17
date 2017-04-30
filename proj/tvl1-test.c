#include <stdio.h>
#include <stdlib.h>
#include <math.h>
//#include "timing.h"
//#include "cl-helper.h"
#include "ppma_io.h"
#include "tvl1-test.h"

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

void freeimg(struct image *img){
    if (img!=NULL){
        free(img->data);
        free(img);
    }
}

struct image *makeimg(int height, int width){
// should this be void makeimg(struct image *img, ...)?
    struct image *retVal = malloc (sizeof (struct image));
    if (retVal == NULL)
        return NULL;

    retVal->data = malloc (height * width * sizeof (double));
    if (retVal->data == NULL){
        free (retVal);
        return NULL;
    }

    retVal->height = height;
    retVal->width  = width;
    return retVal;

}

void nabla(struct image *img, struct image *dx, struct image *dy){
    
    int h, w;
    h = img->height;
    w = img->width;

    for (int i = 0; i < w; i++){
        for (int j = 0; j < h; j++){
            int idx = j*w + i;
            if (i!=w){
                dx->data[idx] -= img->data[idx];
                dx->data[idx] += img->data[idx + 1];
            }
            if (j!=h){
                dy->data[idx] -= img->data[idx];
                dy->data[idx] += img->data[idx + w];
            }
        }
    }
}

void nablaT(struct image *dx, struct image *dy, struct image *img){
    
    int h, w;
    h = img->height;
    w = img->width;
    
    for (int i = 0; i < w; i++){
        for (int j = 0; j < h; j++){
            int idx = j * w + i;
            if (i!=w){
                img->data[idx] -= dx->data[idx];
                img->data[idx + 1] += dx->data[idx];
            }
            if (j!=h){
                img->data[idx] -= dy->data[idx];
                img->data[idx + w] += dy->data[idx];
            }
        }
    }
}

void anorm(struct image *dx, struct image *dy, struct image *a){
    
    int h = dx->height;
    int w = dx->width;
    int sumofsq;

    for (int i = 0; i < h*w; i++){
        sumofsq = pow(dx->data[i], 2) + pow(dy->data[i], 2);
        a->data[i] = sqrt(sumofsq);
    }
}

void project(struct image *dx, struct image *dy, 
             struct image *projx, struct image *projy, 
             double r){

    int h = dx->height;
    int w = dx->width;

    struct image *an;
    an = makeimg(h, w);
    //an->height = h;
    //an->width = w;

    //an->data = malloc(h*w * sizeof (double));
    anorm(dx, dy, an);

    for (int i = 0; i < h*w; i++)
        an->data[i] = (an->data[i]<1.0 ? an->data[i]/r : 1.0);

    for (int i = 0; i < w; i++){
        projx->data[i] = dx->data[i] / an->data[i];
        projy->data[i] = dy->data[i] / an->data[i];
    }
    
    free(an->data);
}

double clip(float n, float low, float high){
    return (n<low ? low: (n>high ? high : n));
}

void shrink(struct image *proj, struct image *img, struct image *sh, double step){

    int h = img->height;
    int w = img->width;

    for (int j = 0; j < h; j++){
        for (int i = 0; i < w; i++){
            int idx = j*w + i;
            sh->data[idx] = proj->data[idx] + clip(img->data[idx] - proj->data[idx], -step, step);
        }
    }
}

void solve_tvl1(struct image *img, struct image *filter, double clambda, int iter){

    double L2 = 8.0;
    double tau = 0.02;
    double theta = 1.0;
    double sigma;
    sigma = 1.0 / (L2 * tau);

    int h, w;

    h = img->height;
    w = img->width;

    struct image *X, *X1, *Px, *Py, *nablaXx, *nablaXy, *nablaTP;
    
    //X->data        = malloc(h * w * sizeof (double));
    //X1->data       = calloc(h * w,  sizeof (double));
    //Px->data       = calloc(h * w,  sizeof (double));
    //Py->data       = calloc(h * w,  sizeof (double));
    //nablaXx->data  = calloc(h * w,  sizeof (double));
    //nablaXy->data  = calloc(h * w,  sizeof (double));
    //nablaTP->data  = calloc(h * w,  sizeof (double));

    X       = makeimg(h, w);
    X1      = makeimg(h, w);
    Px      = makeimg(h, w);
    Py      = makeimg(h, w);
    nablaXx = makeimg(h, w);
    nablaXy = makeimg(h, w);
    nablaTP = makeimg(h, w);
    //X->height = X1->height = Px->height = Py->height = nablaXx->height = nablaXy->height = nablaTP->height = h;
    //X->width  = X1->width  = Px->width  = Py->width  = nablaXx->width  = nablaXy->width  = nablaTP->width  = w;

    for (int i=0; i<h*w; i++)
        X->data[i] = img->data[i];

    nabla(X, Px, Py);
    for (int t = 0; t < iter; t++){
        nabla(X, nablaXx, nablaXy);
        for (int i = 0; i < h*w; i++){
            nablaXx->data[i] *= sigma; 
            nablaXx->data[i] += Px->data[i];
            nablaXy->data[i] *= sigma; 
            nablaXy->data[i] += Py->data[i];
        }
        project(nablaXx, nablaXy, Px, Py, 1.0);
        
        nablaT(Px, Py, nablaTP);
        for (int i = 0; i < h*w; i++){
            nablaTP->data[i] *= -1. * sigma; 
            nablaTP->data[i] += X->data[i];
        }
        shrink(nablaTP, img, X1, clambda*tau);

        for (int i = 0; i < h*w; i++)
            X->data[i] = X1->data[i] + theta * (X1->data[i] - X->data[i]);
    }

    for (int i = 0; i < h*w; i++)
        filter->data[i] = X->data[i];

    free(X);
    free(X1);
    free(Px);
    free(Py);
    free(nablaXx);
    free(nablaXy);
    free(nablaTP);
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

// def make_noisy(int &img, int N):
//     /* add gaussian #noise */
// 	img = np.clip(img + 0.025 * np.random.normal(size=img.shape), 0, 1)
//     # add some outliers in on the right side of the image
//     m = np.random.rand(*img.shape) < 0.2
//     m[:,:300] = 0
//     img[m] = np.random.rand(m.sum())
//     return img

void main(int argc, char *argv[]){

    if(argc != 3)
    {
      fprintf(stderr, "Usage: %s image.ppm num_loops\n", argv[0]);
      abort();
    }
  
    const char* filename = argv[1];
    const int num_loops = atoi(argv[2]);
    int *r, *g, *b;
    int error, xs, ys, rgb_max;
    int xsize, ysize;

    struct image *gray, *img, *filter;

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
    //gray->data   = malloc(xsize * ysize * sizeof (double));
    gray = makeimg(xsize, ysize);
    //filter->data = malloc(xsize * ysize * sizeof (double));
    filter = makeimg(xsize, ysize);

    // --------------------------------------------------------------------------
    // convert image to grayscale
    // --------------------------------------------------------------------------
    for(int n = 0; n < xsize*ysize; ++n) {
//      gray[4*n] = r[n];
//      gray[4*n+1] = g[n];
//      gray[4*n+2] = b[n];
//      gray[4*n+3] = (0.21f*r[n])/rgb_max + (0.72f*g[n])/rgb_max + (0.07f*b[n])/rgb_max;
      gray->data[n] = (0.21f*r[n])/rgb_max + (0.72f*g[n])/rgb_max + (0.07f*b[n])/rgb_max;
    }

    gray->height = ysize;
    gray->width  = xsize;

    solve_tvl1(gray, filter, 1, 101);
    // --------------------------------------------------------------------------
    // output cpu filtered image
    // --------------------------------------------------------------------------
    printf("Writing cpu filtered image\n");
    for(int n = 0; n < xsize*ysize; ++n) {
      r[n] = (int)(filter->data[n] * rgb_max * 0.5);
      g[n] = (int)(filter->data[n] * rgb_max * 0.25);
      b[n] = (int)(filter->data[n] * rgb_max * 1);
    }
    error = ppma_write("output_cpu.ppm", xsize, ysize, r, g, b);
    if(error) { fprintf(stderr, "error writing image"); abort(); }

    free(r);
    free(g);
    free(b);
    free(gray->data);
    free(filter->data);

}
