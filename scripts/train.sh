#!/usr/bin/env bash

OUTPUT_DIR=$1
MODEL_NAME_OR_PATH=$2

set -x
set -e

DIR="$( cd "$( dirname "$0" )" && cd .. && pwd )"
echo "working directory: ${DIR}"

if [ -z "$OUTPUT_DIR" ]; then
  OUTPUT_DIR="${DIR}/checkpoint/"
fi
if [ -z "$MODEL_NAME_OR_PATH" ]; then
  MODEL_NAME_OR_PATH="FacebookAI/roberta-base"
fi
if [ -z "$DATA_DIR" ]; then
  DATA_DIR="${DIR}/data_sample/"
fi
if [ -z "$TRAIN_FILE" ]; then
  TRAIN_FILE="topiocqa_train_sample.jsonl"
fi

mkdir -p "${OUTPUT_DIR}"

PROC_PER_NODE=$(nvidia-smi --list-gpus | wc -l)
torchrun --nproc_per_node ${PROC_PER_NODE} src/train.py \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --per_device_train_batch_size 8 \
    --seed 42 \
    --do_train \
    --train_file "${DATA_DIR}"/"${TRAIN_FILE}" \
    --q_max_len 512 \
    --p_max_len 256 \
    --dataloader_num_workers 0 \
    --num_train_epochs 3 \
    --train_n_passages 16 \
    --learning_rate 2e-5 \
    --gradient_accumulation_steps 1 \
    --warmup_ratio 0.1 \
    --logging_steps 50 \
    --output_dir "${OUTPUT_DIR}" \
    --data_dir "${DATA_DIR}" \
    --save_total_limit 3 \
    --save_strategy epoch \
    --remove_unused_columns False \
    --overwrite_output_dir \
    --disable_tqdm False
