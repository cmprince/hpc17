FLAGS = -std=gnu99 -lm -lrt -Wall -g
TARGETS = int_ring

all: $(TARGETS) 

%: %.c
	mpicc -o $@ $< $(FLAGS)

clean:
	rm $(TARGETS)
