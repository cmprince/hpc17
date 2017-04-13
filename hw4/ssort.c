/* Parallel sample sort
 */
#include <stdint.h>
#include <stdio.h>
#include <unistd.h>
#include <mpi.h>
#include <stdlib.h>
#include <math.h>
#include <limits.h>

// These #defines are used by the randsamp routine below
#define MAX_ALLOC ((uint32_t)0x40000000)  //max allocated bytes, fix per platform
#define MAX_SAMPLES (MAX_ALLOC/sizeof(uint32_t))

int* randsamp(uint32_t x, uint32_t min, uint32_t max);

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
  int *vec, *randindices, *randomsubset, *gather, *splitters, *sendcount, *senddisplace;

  MPI_Init(&argc, &argv);
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  
  //debug loop, attach with gdb --pid <number>
    if(9999999==rank)
    {
        int i = 0;
        char hostname[256];
        gethostname(hostname, sizeof(hostname));
        printf("PID %d on %s ready for attach\n", getpid(), hostname);
        fflush(stdout);
        while (0 == i)
            sleep(5);
    }



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
  randomsubset = malloc(s*   sizeof randomsubset);
  gather       = malloc(s*p* sizeof gather);
  splitters    = malloc(p*   sizeof splitters);
  sendcount    = malloc(p*   sizeof sendcount);
  senddisplace = calloc(p,   sizeof senddisplace);
  /* randomly sample s entries from vector or select local splitters,
   * i.e., every N/P-th entry of the sorted vector */
  randindices = randsamp(s, 0, N-1);
  for (i=0; i<s; ++i){
    randomsubset[i] = vec[randindices[i]];  
  }
  for (i=0; i<p; ++i)
      sendcount[i] = 0;
  /* every processor communicates the selected entries
   * to the root processor; use for instance an MPI_Gather */
  MPI_Gather(&randomsubset, s, MPI_INT, &gather, s, MPI_INT, 0, MPI_COMM_WORLD);

  /* root processor does a sort, determinates splitters that
   * split the data into P buckets of approximately the same size */
  if (0==rank){
    qsort(gather, p*s, sizeof gather, compare);
    int spindex;
    printf("splitters: ");
    for (i=1; i<p; ++i){
      spindex = (int)(p*s*i/(float)p);  
printf("spindex %i: %i,", i-1, spindex);
      splitters[i-1] = gather[spindex];
      printf("%i, ", splitters[i-1]);
    }
    printf("\n");
    splitters[p-1] = INT_MAX;
  }

  /* root process broadcasts splitters */
  MPI_Bcast(&splitters, p, MPI_INT, 0, MPI_COMM_WORLD);

  /* every processor uses the obtained splitters to decide
   * which integers need to be sent to which other processor (local bins) */
  int j=0;
  senddisplace[0] = 0;
  for (i=0; i<N; ++i){
    if (vec[i]<splitters[j]){
      senddisplace[j+1] = i;
    }
    else{
      sendcount[j] = senddisplace[j+1] - senddisplace[j];
      j++;
    }
  }
  for (i=0;i<p;++i)
    printf("j: %i, sc: %i, sd: %i\n", i, sendcount[i], senddisplace[i]);

  /* send and receive: either you use MPI_AlltoallV, or
   * (and that might be easier), use an MPI_Alltoall to share
   * with every processor how many integers it should expect,
   * and then use MPI_Send and MPI_Recv to exchange the data */
  MPI_Alltoallv(MPI_IN_PLACE, sendcount, senddisplace, MPI_INT, 
                &vec, sendcount, senddisplace, MPI_INT, 
                MPI_COMM_WORLD);

  /* do a local sort */
  qsort(vec, N, sizeof(int), compare);

  /* every processor writes its result to a file */

  free(vec);
  MPI_Finalize();
  return 0;
}


/* Random sample without replacement
   Code from http://codegolf.stackexchange.com/questions/4772/random-sampling-without-replacement
   An implementation of Fisher-Yates shuffle.
*/
int* randsamp(uint32_t x, uint32_t min, uint32_t max){
   uint32_t r,i=x,*a;
   if (!x||x>MAX_SAMPLES||x>(max-min+1)) return NULL;
   a=malloc(x*sizeof(uint32_t));
   while (i--) {
      r= (max-min+1-i);
      a[i]=min+=(r ? rand()%r : 0);
      min++;
   }
   while (x>1) {
      r=a[i=rand()%x--];
      a[i]=a[x];
      a[x]=r;
   }
   return a;
}
