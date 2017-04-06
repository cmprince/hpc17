FLAGS = -std=gnu99 -lm -lrt -Wall -g
TARGETS = mpi_solved1 mpi_solved2 mpi_solved3 mpi_solved7

all: $(TARGETS) 

%: %.c
	mpicc -o $@ $< $(FLAGS)

clean:
	rm $(TARGETS)
