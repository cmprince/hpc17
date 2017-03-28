/* Communication ping-pong:
 * Exchange between messages between mpirank
 * 0 <-> 1, 2 <-> 3, ....
 */
#include <stdio.h>
#include <unistd.h>
#include "mpi.h"

int main( int argc, char *argv[])
{
    int rank, numprocs, tag, origin, destination;
    MPI_Status status;
  
    char hostname[1024];
    gethostname(hostname, 1024);
  
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &numprocs);

    int message_out=0;
    int message_in=0;
    int counter=0;
    tag = 99;
    int N =4;
    int R=numprocs;
    int i;

    for (i=0; i<N; i++)
    {
        if(!(i==0 && rank == 0))
        {
            origin = ((rank-1)%R) >= 0 ? ((rank-1)%R) : ((rank-1)%R)+R;
            MPI_Recv(&message_in,  1, MPI_INT, origin, tag, MPI_COMM_WORLD, &status);
            printf("rank %d hosted on %s received from %d the message %d\n", rank, hostname, origin, message_in);
//            counter += message_in;
        }
        counter = message_in + rank;
        if(!(i==(N-1) && rank==(R-1)))
        {
            destination = (rank + 1)%(R);
            MPI_Send(&counter, 1, MPI_INT, destination, tag, MPI_COMM_WORLD);
        }
    }
  
    MPI_Finalize();
    return 0;
}
