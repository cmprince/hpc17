#!/bin/bash

for j in "j" "g"                # jacobi and gauss-seidel
do
    for opt in {0..3}             # optimization level
    do
        for size in 100 1000 10000 100000
        do

            ((num = 0))
            ((result = 0))
            for i in {1..10}         # iterations
            do
                ((num = $(./laplace-iterO$opt $size $j)))
                ((result += num))
                echo $opt $size $i $num >> $j
            done
        done
    done
done
