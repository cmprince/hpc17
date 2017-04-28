#include <stdio.h>
#include <stdlib.h>
#include <math.h>
//#include "timing.h"
#include "cl-helper.h"
#include "ppma_io.h"

#define FILTER_WIDTH 7
#define HALF_FILTER_WIDTH 3

// local size of work group
#define WGX 12
#define WGY 12
#define NON_OPTIMIZED

void nabla(int &img, int &out, int h, int w){
    
    for (int i = 0; i < w; i++){
        for (int j = 0; j < h; j++){
            int idx = j*w + i;
            if (i!=w){
                *out[idx][0] -= *img[idx];
                *out[idx][0] += *img[idx + 1];
            }
            if (j!=h){
                *out[idx][1] -= *img[idx];
                *out[idx][1] += *img[idx + w];
            }
        }
    }

}

void nablaT(int &diff, int &out, int h, int w){
    
    for (int i = 0; i < w; i++){
        for (int j = 0; j < h; j++){
            int idx = j*w + i;
            if (i!=w){
                *out[idx] -= *diff[idx][0];
                *out[idx + 1] += *diff[idx][0];
            }
            if (j!=h){
                *out[idx] -= *diff[idx][1];
                *out[idx + w] += *diff[idx][1];
            }
        }
    }
}

double generateGaussianNoise(const double& mean, const double &stdDev)
{
	static bool hasSpare = false;
 	static double spare;
 
 	if(hasSpare)
 	{
 		hasSpare = false;
 		return mean + stdDev * spare;
 	}
 
 	hasSpare = true;
 	static double u, v, s;
 	do
 	{
 		u = (rand() / ((double) RAND_MAX)) * 2.0 - 1.0;
 		v = (rand() / ((double) RAND_MAX)) * 2.0 - 1.0;
 		s = u * u + v * v;
 	}
 	while( (s >= 1.0) || (s == 0.0) );
 
 	s = sqrt(-2.0 * log(s) / s);
 	spare = v * s;
 	return mean + stdDev * u * s;
}

void make_noisy(int &img, int N){
    for (int i = 0; i<N; ++i){
        img[i] += generateGaussianNoise(0, 0.05); 
        img[i] = min(1, max(0, img[i]));
	}
}

// def make_noisy(int &img, int N):
//     /* add gaussian #noise */
// 	img = np.clip(img + 0.025 * np.random.normal(size=img.shape), 0, 1)
//     # add some outliers in on the right side of the image
//     m = np.random.rand(*img.shape) < 0.2
//     m[:,:300] = 0
//     img[m] = np.random.rand(m.sum())
//     return img

void main(){

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
      r[n] = (int)(congray[4*n] * rgb_max);
      g[n] = (int)(congray[4*n+1] * rgb_max);
      b[n] = (int)(congray[4*n+2] * rgb_max);
    }
    error = ppma_write("output_cpu.ppm", xsize, ysize, r, g, b);
    if(error) { fprintf(stderr, "error writing image"); abort(); }

    free(r);
    free(g);
    free(b);
    free(gray);
    free(tv1);

}
