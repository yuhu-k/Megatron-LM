#!/bin/bash
# This example will start serving the 345M model.
export PYTHONPATH="$PYTHONPATH:$(pwd)"

HOSTNAME=$(hostname)
if [[ $HOSTNAME == *4090* ]]; then
    export NCCL_DEBUG=INFO && export NCCL_SOCKET_IFNAME=enp5s0 && export GLOO_SOCKET_IFNAME=enp5s0
else
    export NCCL_DEBUG=INFO && export NCCL_SOCKET_IFNAME=enp181s0 && export GLOO_SOCKET_IFNAME=enp181s0
fi
MODEL_TYPE="7b"
DISTRIBUTED_ARGS="--nproc_per_node 1 \
                  --nnodes 1 \
                  --rdzv_endpoint eclab40902b:6000 \
		  --rdzv_id 12345 \
		  --rdzv_backend c10d"

CHECKPOINT_DIR="/tmp2/Megatron-LM/llama-2-$MODEL_TYPE-me/hf/tp1-pp1" #"result/$HOSTNAME/2024-06-30_09:49" #
TOKENIZER_MODEL="/tmp2/Megatron-LM/llama-2-$MODEL_TYPE-hf/tokenizer.model"
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
export CUDA_DEVICE_MAX_CONNECTIONS=1

pip install flask-restful

ARGS="
	--num-layers 32 \
	--hidden-size 4096 \
    --num-attention-heads 32 \
"

LLAMA_ARGS="
    --llama-size ${MODEL_TYPE}
"

torchrun $DISTRIBUTED_ARGS tools/run_text_generation_server.py   \
	--tensor-model-parallel-size ${TP} \
	--pipeline-model-parallel-size ${PP} \
	--seq-length 4096 \
	--max-position-embeddings 4096 \
	--tokenizer-type Llama2Tokenizer \
	--tokenizer-model ${TOKENIZER_MODEL} \
	--load ${CHECKPOINT_DIR} \
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
	--llama-size $MODEL_TYPE \
	--swiglu \
	$LLAMA_ARGS
	#$ARGS
	# --encoder-num-layers 32 \
	# --hidden-size 4096 \
	# --num-attention-heads 32
