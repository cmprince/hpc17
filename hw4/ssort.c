/* Parallel sample sort
 */
#include <stdint.h>
#include <stdio.h>
#include <unistd.h>
#include <mpi.h>
#include <stdlib.h>
#include <math.h>
#include <limits.h>
#include "util.h"

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
  int *vec, *randindices, *randomsubset, *gather, *splitters, 
      *sendcount, *recvcount, *senddisplace, *recvdisplace, *vec2;
  timestamp_type t1, t2;

  get_timestamp(&t1);

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
  

  /* Number of random numbers per processor (default or from command line) */
  N = 1000000;
  if (2==argc)
      N = atoi(argv[1]);

  vec = calloc(N, sizeof(int));
  /* seed random number generator differently on every core */
  srand((unsigned int) (rank + 393919));
  /* fill vector with random integers */
  for (i = 0; i < N; ++i) {
    vec[i] = rand();
  }

  /* Get number of processes */
  int p;
  MPI_Comm_size(MPI_COMM_WORLD, &p);

  /* set the subsample size */
  s = (int)sqrt(N);

  randomsubset = malloc(s*   sizeof randomsubset);  
  gather       = malloc(s*p* sizeof gather); //(int)); 
  splitters    = malloc(p*   sizeof splitters); //(int));
  sendcount    = malloc(p*   sizeof sendcount); //(int));
  recvcount    = malloc(p*   sizeof recvcount); //(int));
  senddisplace = calloc(p,   sizeof senddisplace); //(int));
  recvdisplace = calloc(p,   sizeof recvdisplace); //(int));

  /* randomly sample s entries from vector or select local splitters,
   * i.e., every N/P-th entry of the sorted vector */
  randindices = randsamp(s, 0, N-1);
  for (i=0; i<s; ++i){
      randomsubset[i] = vec[randindices[i]]; 
  }
  free(randindices);

  for (i=0; i<p; ++i){
      sendcount[i] = 0;
  }

  /* sort locally */
  qsort(vec, N, sizeof(int), compare);
  
  /* every processor communicates the selected entries
   * to the root processor; use for instance an MPI_Gather */
  MPI_Barrier(MPI_COMM_WORLD);
  MPI_Gather(randomsubset, s, MPI_INT, gather, s, MPI_INT, 0, MPI_COMM_WORLD);
  free(randomsubset);

  /* root processor does a sort, determinates splitters that
   * split the data into P buckets of approximately the same size */
  if (0==rank){
    qsort(gather, p*s, sizeof (int), compare);
    for (i=1; i<p; ++i)
      splitters[i-1] = gather[i*s];
    splitters[p-1] = INT_MAX;
  }
  free(gather);

  /* root process broadcasts splitters */
  MPI_Bcast(splitters, p, MPI_INT, 0, MPI_COMM_WORLD);
  
  /* every processor uses the obtained splitters to decide
   * which integers need to be sent to which other processor (local bins) */
  int j=0;
  senddisplace[0] = 0;

  for (i=0; i<N; ++i){
    if (vec[i]<splitters[j]){
      senddisplace[j+1] = i;
    }
    else{
      j++;
    }
  }
  free(splitters);

  for (i=0;  i<(p-1); ++i){
      sendcount[i] = senddisplace[i+1] - senddisplace[i];
      //printf("rank: %d, j: %i, sc: %i, sd: %i\n", rank, i, sendcount[i], senddisplace[i]);
  }

  sendcount[p-1] = N - senddisplace[p-1];
      //printf("rank: %d, j: %i, sc: %i, sd: %i\n", rank, p-1, sendcount[p-1], senddisplace[p-1]);

  //Send the buffer sizes to the other threads so they know what to receive
  MPI_Alltoall(sendcount, 1, MPI_INT, recvcount, 1, MPI_INT, MPI_COMM_WORLD);

  //Calculate receive displacements and total length of new vec2
  recvdisplace[0]=0;
  int cur=0;
  for (i=1; i<p; ++i){
      cur += recvcount[i-1];
      recvdisplace[i] = cur;
  }
  cur += recvcount[p-1];
  vec2 = calloc(cur, sizeof(int));

  /* debug print 
  for (i=0;  i<(p-1); ++i){
      printf("rank: %d, j: %i, rc: %i, rd: %i\n", rank, i, recvcount[i], recvdisplace[i]);
  }
      printf("rank: %d, j: %i, rc: %i, rd: %i\n", rank, p-1, recvcount[p-1], recvdisplace[p-1]);
  // */

  /* send and receive: either you use MPI_AlltoallV, or
   * (and that might be easier), use an MPI_Alltoall to share
   * with every processor how many integers it should expect,
   * and then use MPI_Send and MPI_Recv to exchange the data */
  MPI_Alltoallv(vec,  sendcount, senddisplace, MPI_INT, 
                vec2, recvcount, recvdisplace, MPI_INT, 
                MPI_COMM_WORLD);
  free(vec);
  free(sendcount);
  free(recvcount);
  free(senddisplace);
  free(recvdisplace);

  /* do a local sort */
  qsort(vec2, cur, sizeof(int), compare);

  /* every processor writes its result to a file */
  char file[1024];
  snprintf(file,1024,"ssort_rank%05d.dat",rank);
  FILE *filePtr = fopen(file,"w");
  for(i=0;i<cur;i++){
    fprintf(filePtr,"%d\n",vec2[i]);
  }
  fclose(filePtr);

  free(vec2);
  
  get_timestamp(&t2);
  double elapsed = timestamp_diff_in_seconds(t1, t2);
  double maxtime = 0;

  //The total time is that of the longest running thread
  MPI_Reduce(&elapsed, &maxtime, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
  if (0==rank) printf("Elapsed time is %.06fs\n", elapsed);

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
