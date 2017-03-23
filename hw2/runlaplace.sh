#!/bin/bash

THISHOST=$(hostname -s)

for executable in "jacobi2D-omp" "gs2D-omp"                # jacobi and gauss-seidel
do
    for size in 50 100 1000
    do

        ((num = 0))
        ((result = 0))
        for i in {1..5}         # iterations
        do
            ((num = $(./$executable $size)))
            ((result += num))
            echo $size $i $num >> ${executable}_${THISHOST}
        done
    done
done
