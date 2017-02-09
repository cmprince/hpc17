#!/bin/bash

for j in "j" "g"                # jacobi and gauss-seidel
do
    for k in {0..3}             # optimization level
    do
        ((num = 0))
        ((result = 0))
        for i in {1..2}         # iterations
        do
            ((num = $(./laplace-iterO$k 1000 $j)))
            ((result += num))
        done
        echo $result
    done
        echo $result
done

echo $result
