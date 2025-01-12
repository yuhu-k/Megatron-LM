#!/bin/bash

export PYTHONPATH="$PYTHONPATH:$(pwd)"
python tools/preprocess_data.py \
    --input wizardlm_orca.jsonl \
    --output wizardlm_orca_dataset \
    --json-keys instruction \
    --json-output-key output \
    --output-prefix output \
    --tokenizer-type Llama2Tokenizer \
    --tokenizer-model /tmp2/Megatron-LM/llama-2-7b-hf/tokenizer.model \
    --workers 4 \
    --append-eod 