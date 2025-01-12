#!/bin/bash


export PYTHONPATH="$PYTHONPATH:$(pwd)"
#MODEL_TYPE="13b-chat"
MODEL_TYPE="70b-chat"

GLOBAL_SIZE=4 # GPUs' num

#if [[ $MODEL_TYPE == 7b* ]]; then
#    TP=1
#elif [[ $MODEL_TYPE == 13b* ]]; then
#    TP=2
#elif [[ $MODEL_TYPE == 70b* ]]; then
#    TP=8
#else
#    echo "MODEL_TYPE 設定有誤: $MODEL_TYPE"
#fi

TP=1
PP=4
VPP=2

#LLAMA_META_FORMAT_DIR="/tmp2/Megatron-LM/llama-2-${MODEL_TYPE}-hf"
# LLAMA_META_FORMAT_DIR="/tmp2/Llama-2-${MODEL_TYPE}-hf"
LLAMA_META_FORMAT_DIR="/home/yuhu/Llama-2-${MODEL_TYPE}-hf"
# MEGATRON_FORMAT_DIR="/home/yuhu/llama-2-${MODEL_TYPE}-me/hf/tp${TP}-pp${PP}-vpp${VPP}"
if [ $VPP -eq 1 ]; then
    MEGATRON_FORMAT_DIR="/tmp2/Megatron-LM/llama-2-${MODEL_TYPE}-me/hf/tp${TP}-pp${PP}"
else
    MEGATRON_FORMAT_DIR="/tmp2/Megatron-LM/llama-2-${MODEL_TYPE}-me/hf/tp${TP}-pp${PP}-vpp${VPP}"
fi
TOKENIZER_MODEL="$LLAMA_META_FORMAT_DIR/tokenizer.model"

if [ -d $MEGATRON_FORMAT_DIR ]; then
    mkdir -p $MEGATRON_FORMAT_DIR
fi

python tools/checkpoint/convert.py --model-type GPT \
   --loader llama_mistral \
   --saver megatron \
   --checkpoint-type hf \
   --model-size llama2-70B \
   --load-dir $LLAMA_META_FORMAT_DIR \
   --save-dir ${MEGATRON_FORMAT_DIR} \
   --tokenizer-model ${TOKENIZER_MODEL} \
   --target-tensor-parallel-size ${TP} \
   --target-pipeline-parallel-size $(($PP * $VPP)) \
   --bf16 \
   --target-virtual-pipeline-size $VPP


