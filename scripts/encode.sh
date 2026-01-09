#!/usr/bin/env bash

INDEX_DIR=$1
MODEL_NAME_OR_PATH=$2

set -x
set -e

DIR="$( cd "$( dirname "$0" )" && cd .. && pwd )"
echo "working directory: ${DIR}"

if [ -z "$INDEX_DIR" ]; then
  INDEX_DIR="${DIR}/index/"
fi
if [ -z "$MODEL_NAME_OR_PATH" ]; then
  MODEL_NAME_OR_PATH="castorini/ance-msmarco-passage"
fi
if [ -z "$DATA_DIR" ]; then
  DATA_DIR="${DIR}/data_sample/"
fi

mkdir -p "${INDEX_DIR}"

PYTHONPATH=src/ python -u src/inferences/encode.py \
    --model_name_or_path $MODEL_NAME_OR_PATH \
    --do_encode \
    --encode_in_path "${DATA_DIR}/collection/passages_sample.jsonl" \
    --encode_save_dir "${INDEX_DIR}" \
    --encode_batch_size 128 \
    --p_max_len 256 \
    --l2_normalize True \
    --dataloader_num_workers 1 \
    --output_dir "${INDEX_DIR}" \
    --data_dir "${DATA_DIR}"