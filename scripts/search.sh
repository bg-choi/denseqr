#!/usr/bin/env bash

SEARCH_FILE=$1
INDEX_DIR=$2
MODEL_NAME_OR_PATH=$3

set -x
set -e

DIR="$( cd "$( dirname "$0" )" && cd .. && pwd )"
echo "working directory: ${DIR}"

DEPTH=50
# by default, search top-200 for train, top-1000 for dev
if [ "${SPLIT}" = "train" ]; then
  DEPTH=50
fi

if [ -z "$MODEL_NAME_OR_PATH" ]; then
  MODEL_NAME_OR_PATH="./checkpoint/"
fi
if [ -z "$OUTPUT_DIR" ]; then
  OUTPUT_DIR="${MODEL_NAME_OR_PATH}"
fi
if [ -z "$DATA_DIR" ]; then
  DATA_DIR="${DIR}/data_sample/"
fi
if [ -z "$INDEX_DIR" ]; then
  INDEX_DIR="${DIR}/index"
fi
if [ -z "$SEARCH_FILE" ]; then
  SEARCH_FILE="topiocqa_dev_sample"
fi
if [ -z "$SEARCH_SIZE" ]; then
    SEARCH_SIZE=32
fi


PYTHONPATH=src/ python -u src/inferences/search.py \
    --fp16 \
    --model_name_or_path "${OUTPUT_DIR}" \
    --search_file "${SEARCH_FILE}" \
    --search_batch_size $SEARCH_SIZE \
    --search_topk "${DEPTH}" \
    --search_out_dir "${OUTPUT_DIR}" \
    --encode_save_dir "${INDEX_DIR}" \
    --q_max_len 512 \
    --l2_normalize True \
    --dataloader_num_workers 1 \
    --output_dir "${OUTPUT_DIR}" \
    --data_dir "${DATA_DIR}"