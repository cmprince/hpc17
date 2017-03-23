PROGRAM = laplace-iter

all:
	for i in 0 1 2 3 ; do \
		$(CC) -o $(PROGRAM)O$$i -std=gnu99 $(PROGRAM).c -lm -lrt -Wall -O$$i; \
	done

clean:
	find -not -name "*.[ch]" -not -name "makefile" -type f -rm
