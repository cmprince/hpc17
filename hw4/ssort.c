/* Parallel sample sort
 */
#include <stdio.h>
#include <unistd.h>
#include <mpi.h>
#include <stdlib.h>
#include <math.h>

static int compare(const void *a, const void *b)
{
  int *da = (int *)a;
  int *db = (int *)b;

  if (*da > *db)
    return 1;
  else if (*da < *db)
    return -1;
  else
    return 0;
}

int main( int argc, char *argv[])
{
  int rank;
  int i, N, s;
  int *vec, *randomsubset, *gather;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);

  /* Number of random numbers per processor (this should be increased
   * for actual tests or could be passed in through the command line */
  N = 100;

  vec = calloc(N, sizeof(int));
  /* seed random number generator differently on every core */
  srand((unsigned int) (rank + 393919));

  /* fill vector with random integers */
  for (i = 0; i < N; ++i) {
    vec[i] = rand();
  }
  printf("rank: %d, first entry: %d\n", rank, vec[0]);

  /* sort locally */
  qsort(vec, N, sizeof(int), compare);

  /* Get number of processes */
  int p;
  MPI_Comm_size(MPI_COMM_WORLD, &p);

  /* set the subsample size */
  s = (int)sqrt(N);
  randomsubset = malloc(s * sizeof randomsubset);
  gather = malloc(s*p* sizeof gather);

  /* randomly sample s entries from vector or select local splitters,
   * i.e., every N/P-th entry of the sorted vector */
  for (i=0; i<s; ++i){
    randomsubset[i] = vec[N*(i/s)]  
  }

  /* every processor communicates the selected entries
   * to the root processor; use for instance an MPI_Gather */
  MPI_Gather(&randomsubset, s, MPI_INTEGER, &gather, s, MPI_INTEGER, 0, MPI_COMM_WORLD);

  /* root processor does a sort, determinates splitters that
   * split the data into P buckets of approximately the same size */
  qsort(gather, p*s, sizeof gather, compare);

  /* root process broadcasts splitters */

  /* every processor uses the obtained splitters to decide
   * which integers need to be sent to which other processor (local bins) */

  /* send and receive: either you use MPI_AlltoallV, or
   * (and that might be easier), use an MPI_Alltoall to share
   * with every processor how many integers it should expect,
   * and then use MPI_Send and MPI_Recv to exchange the data */

  /* do a local sort */

  /* every processor writes its result to a file */

  free(vec);
  MPI_Finalize();
  return 0;
}
