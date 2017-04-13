FLAGS = -std=gnu99 -lm -lrt -g
TARGETS = mpi_solved1 mpi_solved2 mpi_solved3 mpi_solved4 mpi_solved5 mpi_solved6 mpi_solved7 jacobi-mpi2D ssort

all: $(TARGETS) 

%: %.c
	mpicc -o $@ $< $(FLAGS)

clean:
	rm $(TARGETS)
