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

/****************************************
 * tid needs to be private, total shared
 */ 

#pragma omp parallel private(tid) shared(total)
  {
  /* Obtain thread number */
  tid = omp_get_thread_num();
  /* Only master thread does this */
  if (tid == 0) {
    nthreads = omp_get_num_threads();
    printf("Number of threads = %d\n", nthreads);
    }
  printf("Thread %d is starting...\n",tid);

  #pragma omp barrier

  /* do some work */

/***********************************************
 * The update to total below is a data race!
 * Adding a reduction clause fixes this.
 */

  total = 0.0;
  #pragma omp parallel for reduction(+: total) private(i)
  for (i=0; i<1000000; i++) 
     total += i*1.0;
  printf ("Thread %d is done! Total= %e\n",tid,total);

  } /*** End of parallel region ***/
}
