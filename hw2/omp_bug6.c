/******************************************************************************
* FILE: omp_bug6.c
* DESCRIPTION:
*   This program compiles and runs fine, but produces the wrong result.
*   Compare to omp_orphan.c.
* AUTHOR: Blaise Barney  6/05
* LAST REVISED: 06/30/05
******************************************************************************/
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#define VECLEN 100

float a[VECLEN], b[VECLEN];

float dotprod ()
{
int i,tid;
float sum= 0.0;


#pragma omp parallel for reduction(+:sum)
  for (i=0; i < VECLEN; i++)
    {
    //How to avoid repeated calls to this omp function?
    tid = omp_get_thread_num();
    sum = sum + (a[i]*b[i]);
    printf("  tid= %d i=%d\n",tid,i);
    }
  return sum;
}


int main (int argc, char *argv[]) {
int i;
float sum;

for (i=0; i < VECLEN; i++)
  a[i] = b[i] = 1.0 * i;
//sum = 0.0;

//#pragma omp parallel shared(sum)
  sum = dotprod();

printf("Sum = %f\n",sum);

}

