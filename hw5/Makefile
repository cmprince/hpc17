EXECS=multigrid_1d multigrid_2d multigrid_2d_omp
CC=gcc
FLAGS=-O3 -Wall -g -lm -lrt -std=gnu99

all: ${EXECS}

multigrid_1d: multigrid_1d.c
	${CC} $^ -o multigrid_1d ${FLAGS}

multigrid_2d: multigrid_2d.c
	${CC} $^ -o multigrid_2d ${FLAGS}

multigrid_2d_omp: multigrid_2d_omp.c
	${CC} $^ -o multigrid_2d_omp ${FLAGS} -fopenmp

jacobi: jacobi.c
	${CC} $^ -o jacobi ${FLAGS}


clean:
	rm -f ${EXECS}
