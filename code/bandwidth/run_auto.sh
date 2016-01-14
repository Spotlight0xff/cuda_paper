#!/usr/bin/bash

LOOPS=10
NUM=128

rm data${NUM}_* # sorry

# generate data
for count in $(seq 1 $LOOPS)
do
    echo "Round #${count}"
    echo "bandwidth test pageable with ${NUM} samples"
    ./bandwidth -m pageable -n ${NUM} -f plot -o data${NUM}_pageable${count}

    echo "bandwidth test pinned with ${NUM} samples"
    ./bandwidth -m pinned -n ${NUM} -f plot -o data${NUM}_pinned${count}
    echo "\n"
done

# averaging
R --vanilla < avg_data.r

