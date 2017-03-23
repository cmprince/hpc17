/******************************************************************************
* FILE: omp_bug2.c
* DESCRIPTION:
*   Another OpenMP program with a bug. 
* AUTHOR: Blaise Barney 
* LAST REVISED: 04/06/05 
******************************************************************************/
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>

int main (int argc, char *argv[]) 
{
int nthreads, i, tid;
float total;

/*** Spawn parallel region ***/
#pragma omp parallel 
  {
  /* Obtain thread number */
  tid = omp_get_thread_num();
  /* Only master thread does this */
  if (tid == 0) {
    nthreads = omp_get_num_threads();
    printf("Number of threads = %d\n", nthreads);
    total = 0.0;
    }
  printf("Thread %d is starting...\n",tid);

  #pragma omp barrier

  /* do some work */
//  total = 0.0;

/***********************************************
 * The update to total below is a data race!
 * Adding a reduction clause fixes this.
 * original:
 * #pragma omp for schedule(dynamic,10)
 * fixed:
 */
  } /*** End of parallel region ***/
  #pragma omp for schedule(dynamic,10) // reduction(+: total)
  for (i=0; i<1000000; i++) 
#pragma omp atomic
     total = total + i*1.0;
#pragma omp barrier
  printf ("Thread %d is done! Total= %e\n",tid,total);

}
