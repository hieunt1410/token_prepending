#!/bin/bash

# Run retrieval evaluation with Token Prepending
# Usage: bash run_retrieval.sh [config_name]

CONFIG_NAME=${1:-"llama-2-7b-tp-retrieval"}

python evaluate_retrieval.py \
    --config "$CONFIG_NAME" \
    --config_file config.yaml
