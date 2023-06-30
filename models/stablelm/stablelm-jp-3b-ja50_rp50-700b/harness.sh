#!/bin/bash
PROJECT_DIR=""
MODEL_ARGS="pretrained=${PROJECT_DIR}/hf_model/3b-ja50_rp50-700b,tokenizer=${PROJECT_DIR}/tokenizers/nai-hf-tokenizer/,use_fast=False"
TASK="jsquad-1.1-0.2,jcommonsenseqa-1.1-0.2,jnli-1.1-0.2,marc_ja-1.1-0.2"
NUM_FEW_SHOTS="2,3,3,3"
python main.py \
    --model hf-causal \
    --model_args $MODEL_ARGS \
    --tasks $TASK \
    --num_fewshot $NUM_FEW_SHOTS \
    --device "cuda" \
    --output_path "models/stablelm-jp-3b-ja50_rp50-700b/result.json"
