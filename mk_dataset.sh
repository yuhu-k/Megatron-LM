#!/bin/bash

export PYTHONPATH="$PYTHONPATH:$(pwd)"
python tools/preprocess_data.py \
    --input b.json \
    --output dataset2 \
    --output-prefix llama-data \
    --tokenizer-type Llama2Tokenizer \
    --tokenizer-model /tmp2/Megatron-LM/llama-2-7b-hf/tokenizer.model \
    --workers 4 \
    --append-eod 