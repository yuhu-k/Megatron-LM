#!/bin/bash
export PYTHONPATH="$PYTHONPATH:$(pwd)"
export CUDA_DEVICE_MAX_CONNECTIONS=1
export CUDA_VISIBLE_DEVICES=0,1

HOSTNAME=$(hostname)
if [[ $HOSTNAME == *4090* ]]; then
    export NCCL_DEBUG=INFO && export NCCL_SOCKET_IFNAME=eno2 && export GLOO_SOCKET_IFNAME=eno2
else
    export NCCL_DEBUG=INFO && export NCCL_SOCKET_IFNAME=enp181s0 && export GLOO_SOCKET_IFNAME=enp181s0
fi
MODEL_TYPE="7b"
DISTRIBUTED_ARGS="--nproc_per_node 1 \
                  --nnodes 1 \
                  --rdzv_endpoint localhost:6000 \
		  --rdzv_id 12345 \
		  --rdzv_backend c10d"

GLOBAL_SIZE=1 # GPUs' num

# if [[ $MODEL_TYPE == 7b* ]]; then
#     TP=1
# elif [[ $MODEL_TYPE == 13b* ]]; then
#     TP=2
# elif [[ $MODEL_TYPE == 70b* ]]; then
#     TP=8
# else
#     echo "MODEL_TYPE 設定有誤: $MODEL_TYPE"
# 	return
# fi
TP=1
PP=$((${GLOBAL_SIZE} / ${TP}))
echo $PP

# CHECKPOINT_DIR="/tmp2/Megatron-LM/llama-2-${MODEL_TYPE}-me/hf/tp${TP}-pp${PP}" #"results/$HOSTNAME/2024-10-14_19:31"
CHECKPOINT_DIR="/tmp2/Megatron-LM/results/compute2/2025-01-11_15h27m"
TOKENIZER_MODEL="/tmp2/Megatron-LM/llama-2-${MODEL_TYPE}-hf/tokenizer.model"

# pip install flask-restful

LLAMA_ARGS="
    --llama-size ${MODEL_TYPE} \
	--finetune-method "lora"
"
	#  \
	# --finetune-mlp \

torchrun $DISTRIBUTED_ARGS tools/run_text_generation_server.py   \
	--tensor-model-parallel-size ${TP} \
	--pipeline-model-parallel-size ${PP} \
	--seq-length 4096 \
	--max-position-embeddings 4096 \
	--tokenizer-type Llama2Tokenizer \
	--tokenizer-model ${TOKENIZER_MODEL} \
	--load ${CHECKPOINT_DIR} \
	--bf16 \
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
	--micro-batch-size 1 \
	--use-mcore-models \
	--transformer-impl transformer_engine \
	--swiglu \
	$LLAMA_ARGS \
	# --encoder-num-layers 40 \
	# --hidden-size 5120 \
	# --num-attention-heads 40
