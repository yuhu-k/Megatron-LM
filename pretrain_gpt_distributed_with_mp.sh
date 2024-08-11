#!/bin/bash

# Runs the "345M" parameter model

export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_DEBUG=INFO && export NCCL_SOCKET_IFNAME=enp181s0 && export GLOO_SOCKET_IFNAME=enp181s0

GPUS_PER_NODE=2
# Change for multinode config
MASTER_ADDR=eclab3080 # mgmt01
MASTER_PORT=6000
NNODES=2
NODE_RANK=1
WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))
HOSTNAME=$(hostname)

CHECKPOINT_PATH=/tmp2/yuhu/dataset/gpt2_345m
VOCAB_FILE=/tmp2/yuhu/dataset/gpt2-vocab.json
MERGE_FILE=/tmp2/yuhu/dataset/gpt2-merges.txt
DATA_PATH=/tmp2/yuhu/dataset/my-gpt2_text_document
RESULTS_PATH=/tmp2/Megatron-LM/results/$HOSTNAME/$(date '+%Y-%m-%d_%H:%M')

TPdegree=1
PPdegree=4
DPdegree=$(( WORLD_SIZE / TPdegree / PPdegree ))

TENSORBOARD_DIR="./logs/TP_${TPdegree}_PP_${PPdegree}_DP_${DPdegree}"
FILENAME="TP_${TPdegree}_PP_${PPdegree}_DP_${DPdegree}"

DISTRIBUTED_ARGS="
    --nproc_per_node $GPUS_PER_NODE \
    --nnodes $NNODES \
    --rdzv_id 12345 \
    --rdzv_backend c10d \
    --rdzv_endpoint ${MASTER_ADDR}:${MASTER_PORT}
"

GPT_ARGS="
    --tensor-model-parallel-size $TPdegree \
    --pipeline-model-parallel-size $PPdegree \
    --sequence-parallel \
    --num-layers 12 \
    --hidden-size 256 \
    --num-attention-heads 16 \
    --seq-length 1024 \
    --max-position-embeddings 1024 \
    --micro-batch-size 4 \
    --global-batch-size 16 \
    --lr 0.00015 \
    --train-iters 100 \
    --lr-decay-iters 320000 \
    --lr-decay-style cosine \
    --min-lr 1.0e-5 \
    --weight-decay 1e-2 \
    --lr-warmup-fraction .01 \
    --clip-grad 1.0 \
    --fp16 \
    --use-mcore-models
"

NSYS_ARGS="
    --force-overwrite true \
    -o /tmp2/yuhu/$HOSTNAME.log \
    --gpu-metrics-device 0 \
    --capture-range cudaProfilerApi \
    --capture-range-end none
"

PROF_ARGS="
    --profile \
    --profile-ranks 0 1 2 3
"

DATA_ARGS="
    --data-path $DATA_PATH \
    --vocab-file $VOCAB_FILE \
    --merge-file $MERGE_FILE \
    --save $RESULTS_PATH \
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
    --lms-swap-policy dynamic-early
"

if [ -d $RESULTS_PATH ]; then
    mkdir -p $RESULTS_PATH
fi

#nsys profile $NSYS_ARGS \
torchrun $DISTRIBUTED_ARGS pretrain_gpt.py \
    $GPT_ARGS \
    $DATA_ARGS \
    $OUTPUT_ARGS \
    --transformer-impl local \
    #$PROF_ARGS \
    #$LMS_ARGS \
    #--distributed-backend nccl \
    #--save $CHECKPOINT_PATH \
    #--load $CHECKPOINT_PATH

