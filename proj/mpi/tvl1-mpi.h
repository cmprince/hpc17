#include <stdlib.h>
#include <math.h>
//#include "timing.h"
//#include "cl-helper.h"
#include "../ppma_io.h"

//struct image{
//    double *data;  //pixel data
//    int height;
//    int width;
//};
//
//void freeimg(struct image *img);
//struct image *makeimg(int height, int width);
void nabla(double *img, double *dx, double *dy,
           int h, int w, int rank, int xprocs, int yprocs);
void nablaT(double *dx, double *dy, double *img,
            int h, int w, int rank, int xprocs, int yprocs);
void anorm(double *dx, double *dy, double *a, int h, int w); //double *a);
void project(double *dx, double *dy, 
             double *projx, double *projy, 
             double r, double sigma, double *an, int h, int w);
double clip(float n, float low, float high);
void shrink(double *proj, double *img, double *curr, double *sh, 
            double clambda, double tau, double theta, int h, int w);
void solve_tvl1(double *img, double *filter, double clambda, int iter,
                int h, int w, int rank, int xprocs, int yprocs);
void writeimg(double *img, char *fname, int h, int w, double scale, int offset);
