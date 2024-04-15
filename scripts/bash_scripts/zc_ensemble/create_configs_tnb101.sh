#!/bin/bash

searchspaces=(transbench101_micro transbench101_macro)
datasets=(autoencoder class_object class_scene normal jigsaw room_layout segmentsemantic)

for searchspace in "${searchspaces[@]}"
do
    for dataset in "${datasets[@]}"
    do
        scripts/bash_scripts/zc_ensemble/create_configs.sh $searchspace $dataset 9000
    done
done