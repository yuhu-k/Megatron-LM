#!/bin/bash

#source /tmp2/Megatron-LM/.venv/bin/activation
export PYTHONPATH="$PYTHONPATH:$(pwd)"
export CUDA_DEVICE_MAX_CONNECTIONS=1

GPUS_PER_NODE=2
MODEL_TYPE="13b" #"13b"
# Change for multinode config
MASTER_ADDR=eclab40902b # mgmt01
MASTER_PORT=6000
NNODES=1
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))
HOSTNAME=$(hostname)
TIME=$(date '+%Y-%m-%d_%H:%M')

if [[ $HOSTNAME == *4090* ]]; then
    export NCCL_DEBUG=INFO && export NCCL_SOCKET_IFNAME=enp5s0 && export GLOO_SOCKET_IFNAME=enp5s0
else
    export NCCL_DEBUG=INFO && export NCCL_SOCKET_IFNAME=enp181s0 && export GLOO_SOCKET_IFNAME=enp181s0
fi

if [[ $MODEL_TYPE == 7b* ]]; then
    # TPdegree=1
    LAYER_NUM=32
elif [[ $MODEL_TYPE == 13b* ]]; then
    # TPdegree=2
    LAYER_NUM=40
elif [[ $MODEL_TYPE == 70b* ]]; then
    # TPdegree=8
    LAYER_NUM=40
else
    echo "MODEL_TYPE 設定有誤: $MODEL_TYPE"
	return
fi

TPdegree=1
PPdegree=2
DPdegree=$(( WORLD_SIZE / TPdegree / PPdegree ))

CHECKPOINT_PATH=llama-2-${MODEL_TYPE}-me/hf/tp${TPdegree}-pp${PPdegree}
DATA_PATH=/tmp2/Megatron-LM/dataset/llama-data_text_document #/tmp2/yuhu/dataset/my-gpt2_text_document
RESULTS_PATH=/tmp2/Megatron-LM/results/$HOSTNAME/$TIME
TOKENIZER_MODEL=/tmp2/Megatron-LM/llama-2-$MODEL_TYPE-hf/tokenizer.model

TENSORBOARD_DIR="./logs/TP_${TPdegree}_PP_${PPdegree}_DP_${DPdegree}"
FILENAME="TP_${TPdegree}_PP_${PPdegree}_DP_${DPdegree}"

if [[ $HOSTNAME == "eclab3080" ]]; then
    NODE_RANK=0
elif [[ $HOSTNAME == mgmt01 ]]; then
    NODE_RANK=3
elif [[ $HOSTNAME == *40902 ]]; then
    NODE_RANK=2
elif [[ $HOSTNAME == *40902b ]]; then
    NODE_RANK=1
else
    echo "Error, no such hostname '$HOSTNAME'"
    exit
fi

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --rdzv_id 12345 \
    --rdzv_backend c10d \
    --rdzv_endpoint ${MASTER_ADDR}:${MASTER_PORT} \
    --node-rank $NODE_RANK \
"

GPT_ARGS="
    --tensor-model-parallel-size $TPdegree \
    --pipeline-model-parallel-size $PPdegree \
    --seq-length 1024 \
    --max-position-embeddings 1024 \
    --tokenizer-type Llama2Tokenizer \
    --tokenizer-model ${TOKENIZER_MODEL} \
    --micro-batch-size 1 \
    --global-batch-size 4 \
    --lr 0.00015 \
    --train-iters 100 \
    --lr-decay-iters 320000 \
    --lr-decay-style cosine \
    --min-lr 1.0e-5 \
    --weight-decay 1e-2 \
    --lr-warmup-fraction .01 \
    --clip-grad 1.0 \
    --fp16 \
    --use-mcore-models \
    --exit-on-missing-checkpoint \
    --use-checkpoint-args \
	--no-load-optim \
	--no-load-rng \
	--untie-embeddings-and-output-weights \
	--use-rotary-position-embeddings \
	--normalization RMSNorm \
	--position-embedding-type rope \
	--no-masked-softmax-fusion \
	--attention-softmax-in-fp32 \
    --train-iters 100 \
"

#--num-layers-per-virtual-pipeline-stage $(($LAYER_NUM / $PPdegree))

LLAMA_ARGS="
    --llama-size ${MODEL_TYPE} \
    --finetune-method lora \
    --swiglu
"

NSYS_ARGS="
    --force-overwrite true \
    -o /tmp2/yuhu/${HOSTNAME}_${TIME} \
    --gpu-metrics-device all \
    --capture-range cudaProfilerApi \
    --capture-range-end stop-shutdown \
    --trace=nvtx,cuda,cudnn \
    --backtrace=none \
    --gpu-metrics-set 0
"

PROF_ARGS="
    --profile \
    --profile-ranks 0 1 2 3 4 5 6 7\
    --profile-step-start 5 \
    --profile-step-end 10 \
    --profile-output /tmp2/yuhu/${HOSTNAME}_${MODEL_TYPE}_${TIME}.json
"

DATA_ARGS="
    --data-path $DATA_PATH \
    --split 949,50,1
"
OUTPUT_ARGS="
    --log-interval 100 \
    --save-interval 10000 \
    --eval-interval 1000 \
    --eval-iters 10
"

LMS_ARGS="
    --lms \
    --lms-iterations 3 \
    --lms-path $TENSORBOARD_DIR \
    --lms-filename $FILENAME \
    --lms-profile \
    --lms-swap \
    --lms-swap-policy dynamic-early \
"
if [ -d $RESULTS_PATH ]; then
    mkdir -p $RESULTS_PATH
fi

echo "nsys profile $NSYS_ARGS \
    torchrun $DISTRIBUTED_ARGS finetune_llama.py \
    $GPT_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    $LLAMA_ARGS \
    --transformer-impl transformer_engine \
    --save $RESULTS_PATH \
    --load $CHECKPOINT_PATH \
    $PROF_ARGS \
    --recompute-method uniform \
    --recompute-granularity full \
    --recompute-num-layers $(($LAYER_NUM / $PPdegree))
"

nsys profile $NSYS_ARGS \
torchrun $DISTRIBUTED_ARGS finetune_llama.py \
    --distributed-backend nccl \
    $GPT_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    $LLAMA_ARGS \
    --transformer-impl transformer_engine \
    --save $RESULTS_PATH \
    --load $CHECKPOINT_PATH \
    $PROF_ARGS \
    $LMS_ARGS \
    --swap-weight
    
