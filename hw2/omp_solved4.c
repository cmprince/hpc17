/******************************************************************************
* FILE: omp_bug4.c
* DESCRIPTION:
*   This very simple program causes a segmentation fault.
* AUTHOR: Blaise Barney  01/09/04
* LAST REVISED: 04/06/05
******************************************************************************/
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#define N 1048

int main (int argc, char *argv[]) 
{
int nthreads, tid, i, j;

/**************
 * This is a pretty big size for the stack...

double a[N][N];
 
 * But we can't malloc here, because the malloc only allocates memory blocks
 * in the serial (that is, not parallel) scope. So let's do it in the
 * parallel region instead:
 * *************/


/* Fork a team of threads with explicit variable scoping */
#pragma omp parallel shared(nthreads) private(i,j,tid) //,a)
  {

// Do the malloc here (**a to use double indexing)
double **a;
a = malloc(N * sizeof *a);

// Now every thread has its own privately malloc'd space for a[].

if (a)
    for (i = 0; i<N; i++)
        a[i] = malloc(N * sizeof *a[i]);

/* Obtain/print thread info */
  tid = omp_get_thread_num();
  if (tid == 0) 
    {
    nthreads = omp_get_num_threads();
    printf("Number of threads = %d\n", nthreads);
    }
  printf("Thread %d starting...\n", tid);

  /* Each thread works on its own private copy of the array */
  for (i=0; i<N; i++)
    for (j=0; j<N; j++)
      a[i][j] = tid + i + j;

  /* For confirmation */
  printf("Thread %d done. Last element= %f\n",tid,a[N-1][N-1]);

  //Free our memory from the heap. Need to do this in the parallel region
  //so that each thread takes care of its own private storage.
  
  for (i = 0; i<N; i++) 
      free (a[i]);
  free (a);
  }  /* All threads join master thread and disband */

}
