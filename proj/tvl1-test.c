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

void anorm(struct image *img, double *a){
    
    int h = img->height;
    int w = img->width;
    int sumofsq;

    for (int j = 0; j < h; j++){
        sumofsq = 0;
        for (int i = 0; i < w; i++){
            int idx = j*w + i;
            sumofsq += pow(img->data[idx], 2);
        }
        a[j] = sqrt(sumofsq);
    }
}

void project(struct image *img, struct image *proj, double r){

    int h = img->height;
    int w = img->width;

    double *an = malloc(h * sizeof an);
    anorm(img, an);

    for (int i = 0; i < h; i++)
        an[i] = (an[i]<1.0 ? an[i]/r : 1.0);

    for (int j = 0; j < h; j++){
        for (int i = 0; i < w; i++){
            int idx = j * w + i;
            proj->data[idx] = img->data[idx] / an[j];
        }
    }
    free(an);
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
    float *gray, *tv1;
    int *r, *g, *b;
    int error, xs, ys, rgb_max;
    int xsize, ysize;

    // --------------------------------------------------------------------------
    // load image
    // --------------------------------------------------------------------------
    printf("Reading ``%s''\n", filename);
    ppma_read(filename, &xsize, &ysize, &rgb_max, &r, &g, &b);
    printf("Done reading ``%s'' of size %dx%d\n", filename, xsize, ysize);
  

    // --------------------------------------------------------------------------
    // allocate CPU buffers
    // --------------------------------------------------------------------------
    posix_memalign((void**)&gray, 32, xsize*ysize*sizeof(float));
    //posix_memalign((void**)&gray, 32, 4*xsize*ysize*sizeof(float));
    if(!gray) { fprintf(stderr, "alloc gray"); abort(); }
    posix_memalign((void**)&tv1, 32, 4*xsize*ysize*sizeof(float));
    if(!tv1) { fprintf(stderr, "alloc gray"); abort(); }
    //posix_memalign((void**)&congray, 32, 4*xsize*ysize*sizeof(float));
    //if(!congray) { fprintf(stderr, "alloc gray"); abort(); }
    //posix_memalign((void**)&congray_cl, 32, 4*xsize*ysize*sizeof(float));
    //if(!congray_cl) { fprintf(stderr, "alloc gray"); abort(); }

  
    // --------------------------------------------------------------------------
    // convert image to grayscale
    // --------------------------------------------------------------------------
    for(int n = 0; n < xsize*ysize; ++n) {
//      gray[4*n] = r[n];
//      gray[4*n+1] = g[n];
//      gray[4*n+2] = b[n];
//      gray[4*n+3] = (0.21f*r[n])/rgb_max + (0.72f*g[n])/rgb_max + (0.07f*b[n])/rgb_max;
      gray[n] = (0.21f*r[n])/rgb_max + (0.72f*g[n])/rgb_max + (0.07f*b[n])/rgb_max;
    }
  
  
    // --------------------------------------------------------------------------
    // output cpu filtered image
    // --------------------------------------------------------------------------
    printf("Writing cpu filtered image\n");
    for(int n = 0; n < xsize*ysize; ++n) {
      r[n] = (int)(gray[4*n] * rgb_max);
      g[n] = (int)(gray[4*n+1] * rgb_max);
      b[n] = (int)(gray[4*n+2] * rgb_max);
    }
    error = ppma_write("output_cpu.ppm", xsize, ysize, r, g, b);
    if(error) { fprintf(stderr, "error writing image"); abort(); }

    free(r);
    free(g);
    free(b);
    free(gray);
    free(tv1);

}
