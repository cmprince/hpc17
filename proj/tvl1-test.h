#include <stdlib.h>
#include <math.h>
//#include "timing.h"
//#include "cl-helper.h"
#include "ppma_io.h"

struct image{
    double *data;  //pixel data
    int height;
    int width;
};

void nabla(struct image *img, struct image *dx, struct image *dy);
void nablaT(struct image *dx, struct image *dy, struct image *img);
void anorm(struct image *dx, struct image *dy, struct image *a);
void project(struct image *dx, struct image *dy, 
             struct image *projx, struct image *projy, 
             double r);
double clip(float n, float low, float high);
void shrink(struct image *proj, struct image *img, struct image *sh, double step);
void solve_tvl1(struct image *img, struct image *filter, double clambda, int iter);
