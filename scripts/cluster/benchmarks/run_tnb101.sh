#!/bin/bash

experiment=$1
predictor=$2
start_seed=9000

if [ -z "$experiment" ]
then
    echo "Experiment argument not provided"
    exit 1
fi

if [ -z "$predictor" ];
then
    predictors=(zen) #fisher  grad_norm grasp jacov snip synflow epe_nas flops params plain l2_norm nwot)
    memory=(    32G)  #64G     32G       64G   32G   32G  32G     32G     5G    5G     32G   32G     32G)
else
    predictors=($predictor)
fi

searchspaces=(transbench101_micro) # transbench101_macro)
datasets=(autoencoder class_object class_scene normal jigsaw room_layout segmentsemantic)

start=0
end=4095
n_models=1000
range=${start}-${end}:${n_models}

sed -i "s/JOB_ARRAY_RANGE/$range/" ./scripts/cluster/benchmarks/run.sh
sed -i "s/JOB_N_MODELS/$n_models/" ./scripts/cluster/benchmarks/run.sh

for searchspace in "${searchspaces[@]}"
do
    for dataset in "${datasets[@]}"
    do
        for pred_index in "${!predictors[@]}"
        do
            pred="${predictors[$pred_index]}"
            mem="${memory[$pred_index]}"

            if [ "$pred" ==  "fisher" -a "$dataset" == "jigsaw" ]; then
                echo "Found $pred-$dataset combination. Skipping it."
                continue
            fi

            sed -i "s/THE_JOB_NAME/${searchspace}-${dataset}-${pred}/" ./scripts/cluster/benchmarks/run.sh
            sed -i "s/MEM_FOR_JOB/$mem/" ./scripts/cluster/benchmarks/run.sh

            echo $pred $dataset
            sbatch ./scripts/cluster/benchmarks/run.sh $searchspace $dataset $pred $start_seed $experiment --bosch
            # cat ./scripts/cluster/benchmarks/run.sh

            # Restore the placeholders in run.sh
            sed -i "s/${searchspace}-${dataset}-${pred}/THE_JOB_NAME/" ./scripts/cluster/benchmarks/run.sh
            sed -i "s/#SBATCH --mem=${mem}/#SBATCH --mem=MEM_FOR_JOB/" ./scripts/cluster/benchmarks/run.sh

            sed -i "s/x.mem${mem}.%A-%a.%N.out/x.memMEM_FOR_JOB.%A-%a.%N.out/" ./scripts/cluster/benchmarks/run.sh
            sed -i "s/x.mem${mem}.%A-%a.%N.err/x.memMEM_FOR_JOB.%A-%a.%N.err/" ./scripts/cluster/benchmarks/run.sh
        done

        echo ""
    done
done

# Restore placeholders
sed -i "s/#SBATCH -a $range/#SBATCH -a JOB_ARRAY_RANGE/" ./scripts/cluster/benchmarks/run.sh
sed -i "s/N_MODELS=$n_models/N_MODELS=JOB_N_MODELS/" ./scripts/cluster/benchmarks/run.sh
