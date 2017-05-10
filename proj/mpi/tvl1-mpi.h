#include <stdlib.h>
#include <math.h>
//#include "timing.h"
//#include "cl-helper.h"
#include "../ppma_io.h"
#include <inttypes.h>

//struct image{
//    double *data;  //pixel data
//    int height;
//    int width;
//};
//
//void freeimg(struct image *img);
//struct image *makeimg(int height, int width);
void nabla(double *img, double *dx, double *dy,
           uint32_t h, uint32_t w, int rank, uint32_t xprocs, uint32_t yprocs);
void nablaT(double *dx, double *dy, double *img,
            uint32_t h, uint32_t w, int rank, uint32_t xprocs, uint32_t yprocs);
void anorm(double *dx, double *dy, double *a, uint32_t h, uint32_t w); //double *a);
void project(double *dx, double *dy, 
             double *projx, double *projy, 
             double r, double sigma, double *an, uint32_t h, uint32_t w);
double clip(float n, float low, float high);
void shrink(double *proj, double *img, double *curr, double *sh,
            double clambda, double tau, double theta, uint32_t h, uint32_t w);
void solve_tvl1(double *img, double *filter, double clambda, int iter,
                uint32_t h, uint32_t w, int rank, uint32_t xprocs, uint32_t yprocs);
void writeimg(double *img, char *fname, int h, int w, double scale, int offset);
