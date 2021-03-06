/* Communication ping-pong:
 * Exchange between messages between mpirank
 * 0 <-> 1, 2 <-> 3, ....
 */
#include <stdio.h>
#include <unistd.h>
#include <stdlib.h>
#include <math.h>
#include "mpi.h"
#include "util.h"

int main( int argc, char *argv[])
{
    int rank, R, N, tag, origin, destination;
    
    MPI_Status status;
  
    char hostname[1024];
    gethostname(hostname, 1024);
 
    if (argc!=2){
        printf("Usage: mpirun -np R %s n\n"
               "    R: number of processes\n"
               "    n: number of loops of the MPI ring.\n", argv[0]);
        exit(0);
    }
    else
        N = atoi(argv[1]);
    
    timestamp_type time1, time2;
    get_timestamp(&time1);

    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &R);

    int message_out=0;
    int message_in=0;
    int counter=0;
    tag = 99;
    int i;
    const int twomeg = pow(2,18);

    //int *a = malloc(2*pow(2,20));
    char a[twomeg];

    for (i=0; i<N; i++)
    {
        if(!(i==0 && rank == 0))
        {
            origin = ((rank-1)%R) >= 0 ? ((rank-1)%R) : ((rank-1)%R)+R;
            MPI_Recv(&a,  twomeg, MPI_BYTE, origin, tag, MPI_COMM_WORLD, &status);
//            printf("rank %d hosted on %s received from %d the message %d\n", rank, hostname, origin, message_in);
//            counter += message_in;
        }
        if(!(i==(N-1) && rank==(R-1)))
        {
            destination = (rank + 1)%(R);
            MPI_Send(&a, twomeg, MPI_BYTE, destination, tag, MPI_COMM_WORLD);
        }
    }
  
    MPI_Finalize();
    get_timestamp(&time2);                                                                  
    double elapsed = timestamp_diff_in_seconds(time1,time2);
    
    if (rank==R-1){
        printf("Time elapsed is %f seconds.\n", elapsed);
        printf("%f ns latency\n", elapsed*1e9/(float)(N*R-2));
        printf("%f GB/s\n", 2*(N*R-2)/elapsed/pow(2,10));
    }
    //printf("Inner product is %f.\n", prod);
  


    return 0;
}
