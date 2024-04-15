#!/bin/bash

searchspace=nasbench101
datasets=(cifar10)

for dataset in "${datasets[@]}"
do
    scripts/bash_scripts/benchmarks/create_configs.sh $searchspace $dataset 9000
done