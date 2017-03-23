FLAGS = -fopenmp -lm -lrt -Wall
TARGETS = omp_solved2 omp_solved3 omp_solved4 omp_solved5 omp_solved6 jacobi2D-omp gs2D-omp

all: $(TARGETS) 

%: %.c
	$(CC) -o $@ $< $(FLAGS)

clean:
	rm $(TARGETS)
