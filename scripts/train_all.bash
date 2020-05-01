#!/bin/bash
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
gpu_count=$(nvidia-smi --query-gpu=count --format=csv,noheader)
gpu_seq=$(seq -s, ${gpu_count})

train_dataset() {
    obs_len=${1-8}
    pred_len=${2-12}
    model_dir=${3-my-models}
    mkdir -p ${model_dir}
    find ./datasets/* -prune -type d | while IFS= read -r dir; do
        name=`echo "$dir" | sed "s@^.*/\(.*\)@\1@g"`
        echo $name
        /usr/bin/time -v -o ./my-models/sgan_${name}_${obs_len}_${pred_len}-timestamp.txt \
            echo python2 scripts/train.py \
            --print_every=1 \
            --checkpoint_every=1 \
            --output_dir=./my_models \
            --checkpoint_name=sgan_${name}_${obs_len}_${pred_len} \
            --obs_len=${obs_len} \
            --pred_len=${pred_len} \
            --gpu_num=${gpu_count}
    done
}

pushd "${DIR}/.."
mkdir -p my-models
train_dataset 8 8
train_dataset 8 12
popd
