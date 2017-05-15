EXECS=jacobi multigrid_1d
CC=gcc
FLAGS=-O3 -Wall -g -lm

all: ${EXECS}

multigrid_1d: multigrid_1d.c
	${CC} $^ -o multigrid_1d ${FLAGS}

jacobi: jacobi.c
	${CC} $^ -o jacobi ${FLAGS}


clean:
	rm -f ${EXECS}
