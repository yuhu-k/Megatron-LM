#!/bin/bash


export PYTHONPATH="$PYTHONPATH:$(pwd)"
MODEL_TYPE="13b"

GLOBAL_SIZE=2 # GPUs' num

if [[ $MODEL_TYPE == 7b* ]]; then
    TP=1
elif [[ $MODEL_TYPE == 13b* ]]; then
    TP=2
elif [[ $MODEL_TYPE == 70b* ]]; then
    TP=8
else
    echo "MODEL_TYPE 設定有誤: $MODEL_TYPE"
fi
TP=1
PP=$((${GLOBAL_SIZE} / ${TP}))

LLAMA_META_FORMAT_DIR="llama-2-${MODEL_TYPE}-hf"
MEGATRON_FORMAT_DIR="llama-2-${MODEL_TYPE}-me/hf/tp${TP}-pp${PP}"
TOKENIZER_MODEL="$LLAMA_META_FORMAT_DIR/tokenizer.model"

if [ -d $MEGATRON_FORMAT_DIR ]; then
    mkdir -p $MEGATRON_FORMAT_DIR
fi

python tools/checkpoint/convert.py --model-type GPT \
   --loader llama_mistral \
   --saver megatron \
   --checkpoint-type hf \
   --model-size llama2-$(echo $MODEL_TYPE | tr '[:lower:]' '[:upper:]' | cut -d '-' -f 1) \
   --load-dir $LLAMA_META_FORMAT_DIR \
   --save-dir ${MEGATRON_FORMAT_DIR} \
   --tokenizer-model ${TOKENIZER_MODEL} \
   --target-tensor-parallel-size ${TP} \
   --target-pipeline-parallel-size ${PP}