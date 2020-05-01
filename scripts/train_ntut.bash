#!/bin/bash
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
gpu_count=$(nvidia-smi --query-gpu=count --format=csv,noheader)
gpu_seq=$(seq -s, ${gpu_count})

pushd "${DIR}/.."
# obs_len: $1
# pred_len: $2
# name: $3

name=${1-ntut_library}
obs_len=${2-8}
pred_len=${3-12}
gpu_num=${4-0}
model_dir="models/sgan-${name}-${obs_len}-${pred_len}"

mkdir -p ${model_dir}
# name=`echo "$dir" | sed "s@^.*/\(.*\)@\1@g"`
name=ntut_library
/usr/bin/time -v -o "${model_dir}/timestamp-$(date +%Y%m%d%H%M%S%z).txt" \
    python2 scripts/train.py \
    --print_every=1 \
    --checkpoint_every=1 \
    --output_dir="${model_dir}" \
    --checkpoint_name=sgan-${name}-${obs_len}-${pred_len} \
    --obs_len="${obs_len}" \
    --pred_len="${pred_len}" \
    --dataset_name="${name}" \
    --gpu_num="${gpu_num}" \
    --delim="," \
    --d_type='local' \
    --pred_len=8 \
    --encoder_h_dim_g=32 \
    --encoder_h_dim_d=64 \
    --decoder_h_dim=32 \
    --embedding_dim=16 \
    --bottleneck_dim=32 \
    --mlp_dim=64 \
    --num_layers=1 \
    --noise_dim=8 \
    --noise_type=gaussian \
    --noise_mix_type=global \
    --pool_every_timestep=0 \
    --l2_loss_weight=1 \
    --batch_norm=0 \
    --dropout=0 \
    --batch_size=32 \
    --g_learning_rate=1e-3 \
    --g_steps=1 \
    --d_learning_rate=1e-3 \
    --d_steps=2 \
    --checkpoint_every=10 \
    --print_every=50 \
    --num_iterations=20000 \
    --num_epochs=500 \
    --pooling_type='pool_net' \
    --clipping_threshold_g=1.5 \
    --best_k=10 \
    --restore_from_checkpoint=1
popd
